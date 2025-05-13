import os
from argparse import ArgumentParser, Namespace
from tqdm.auto import tqdm

import einops

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import distributed as dist
from torchmetrics.classification import MulticlassJaccardIndex

from mdistiller.dataset.ade_20k import get_ade_20k_dataloaders
from mdistiller.models._base import ModelBase
from mdistiller.models import imagenet_model_dict

from tools.lineval.utils import (
    init_parser,
    prepare_lineval_dir,
    load_from_checkpoint,
)
from mdistiller.utils import dist_fn


# Utility

class SemSegHead(nn.Module):
    def __init__(self, embed_dim: int, upsample_factor: int=4, num_classes: int=150):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.num_classes = num_classes
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor):
        """
        feat: (B, N+?, C), ViT final layer output with CLS token
        Returns:
            pred_logit: (B, num_bins, H, W)
        """
        resolution = int(feat.size(1) ** 0.5)
        feat = feat[:, -resolution*resolution:]  # (B, N, C)

        feat = einops.rearrange(feat, 'B (H W) C -> B C H W', H=resolution)
        logit = self.head(feat)  # (B, cls, H, W)
        upsampled = F.interpolate(
            logit,
            scale_factor=self.upsample_factor,
            mode='bilinear',
            align_corners=False,
        )  # (B, cls, H*r, W*r)
        return upsampled


def main(args: Namespace):
    rank = int(os.environ['LOCAL_RANK'])
    IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
    DEVICE = rank
    EPOCHS = args.epochs
    
    if IS_MASTER:
        _, log_filename, best_filename, last_filename = prepare_lineval_dir(
            args.expname, 
            tag=str(args.tag), 
            dataset='ade20k', 
            args=vars(args)
        )
    
    # DataLoaders, Models
    train_loader, test_loader, _ = get_ade_20k_dataloaders(
        args.batch_size//world_size, args.test_batch_size//world_size,
        args.num_workers, use_ddp=True,
    )
    if args.timm_model is not None:
        print(f"Loading {args.timm_model} from timm")
        model = imagenet_model_dict[args.timm_model](pretrained=True)
    else:
        model, _ = load_from_checkpoint(args.expname, tag=args.tag)
    model: ModelBase = model.cuda(DEVICE)
    head = SemSegHead(model.embed_dim, upsample_factor=16).cuda(DEVICE)
    
    for param in head.parameters():
        param.data = param.data.contiguous()
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    head = nn.parallel.DistributedDataParallel(head, device_ids=[rank])
    miou = MulticlassJaccardIndex(150, ignore_index=-1).cuda(rank)
    optimizer = optim.SGD(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=EPOCHS*len(train_loader),
        eta_min=1.0E-8,
    )
    
    # Training Loop
    best_miou = -torch.inf
    train_loss_list, train_miou_list, test_loss_list, test_imou_list = [], [], [], []
    for epoch in range(args.epochs):
        
        with tqdm(train_loader, desc=f'TRAIN {epoch+1}', dynamic_ncols=True, disable=not IS_MASTER) as bar:
            total_loss, total_miou, total, total_classes = 0, 0, 0, 0
            for input, target in bar:
                target = target.long()
                with torch.no_grad():
                    x = model.forward(input.cuda(DEVICE))[1]['feats'][-1]
                pred_logit = head.forward(x)  # (B, 150, H, W)

                loss = F.cross_entropy(pred_logit, target.cuda(DEVICE), ignore_index=-1)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                with torch.no_grad():
                    loss = torch.nn.functional.cross_entropy(pred_logit, target.to(DEVICE), ignore_index=-1, reduction='none')
                    loss_all = dist_fn.gather(loss)
                    local_miou = miou(pred_logit, target.to(DEVICE))[None]
                    miou_all = dist_fn.gather(local_miou)
                    
                    total_loss += loss_all.mean().cpu().item()
                    total_miou += miou_all.sum().cpu().item()
                    total += loss_all.size(0)
                    total_classes += miou_all.numel()
                    
                    if IS_MASTER:
                        bar.set_postfix(dict(
                            lr=optimizer.param_groups[0]['lr'],
                            loss=total_loss/total,
                            miou=f'{total_miou/total_classes*100:.2f}%',
                        ))
            train_miou = total_miou / total_classes
            train_loss = total_loss / total

        with tqdm(test_loader, desc=f' TEST {epoch+1}', dynamic_ncols=True, disable=not IS_MASTER) as bar, torch.no_grad():
            total_loss, total_miou, total, total_classes = 0, 0, 0, 0
            for input, target in bar:
                target = target.long()
                x = model.forward(input.cuda(DEVICE))[1]['feats'][-1]
                pred_logit = head.forward(x) # (B, 150, H, W)
                
                loss = torch.nn.functional.cross_entropy(pred_logit, target.to(DEVICE), ignore_index=-1, reduction='none')
                loss_all = dist_fn.gather(loss)
                local_miou = miou(pred_logit, target.to(DEVICE))[None]
                miou_all = dist_fn.gather(local_miou)
                
                total_loss += loss_all.mean().cpu().item()
                total_miou += miou_all.sum().cpu().item()
                total += loss_all.size(0)
                total_classes += miou_all.numel()
                
                if IS_MASTER:
                    bar.set_postfix(dict(
                        loss=total_loss/total,
                        miou=f'{total_miou/total_classes*100:.2f}%',
                    ))
            test_miou = total_miou / total
            test_loss = total_loss / total
        
        # Logging
        train_loss_list.append(train_loss)
        train_miou_list.append(train_miou)
        test_loss_list.append(test_loss)
        test_imou_list.append(test_miou)
        
        if IS_MASTER:
            with open(log_filename, 'a') as file:
                print(f'- epoch: {epoch+1}', file=file)
                print(f'  train_loss: {train_loss:.4f}', file=file)
                print(f'  train_miou: {train_miou*100:.4f} %', file=file)
                print(f'  test_loss: {test_loss:.4f}', file=file)
                print(f'  test_miou: {test_miou*100:.4f} %', file=file)
                print(file=file)
            
            ckpt = dict(
                epoch=epoch+1,
                train_loss=train_loss_list,
                train_rmse=train_miou_list,
                test_loss=test_loss_list,
                test_rmse=test_imou_list,
                head={
                    key: val.clone().detach().cpu()
                    for key, val in head.state_dict().items()
                },
            )
            if test_miou > best_miou:
                best_miou = test_miou
                torch.save(ckpt, str(best_filename))
            torch.save(ckpt, str(last_filename))


if __name__ == '__main__':
    parser = ArgumentParser('lineval.ade20k')
    init_parser(parser, defaults=dict(epochs=1000))
    args = parser.parse_args()
    
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['IS_MASTER_NODE'] = str(int(rank == 0))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    finally:
        dist.destroy_process_group()
