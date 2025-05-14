import os
from argparse import ArgumentParser, Namespace
from tqdm.auto import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import distributed as dist

from mdistiller.dataset.imagenet import get_imagenet_dataloaders
from mdistiller.models._base import ModelBase
from mdistiller.models import imagenet_model_dict

from tools.lineval.utils import (
    init_parser,
    prepare_lineval_dir,
    load_from_checkpoint,
)
from mdistiller.utils import dist_fn


# Utility

class ClsHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 1000):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, feat: torch.Tensor):
        """
        feat: (B, N+?, C), ViT final layer output with CLS token
        Returns:
            pred_logit: (B, num_classes)
        """
        cls_token = feat[:, 0]
        pred = self.head(cls_token)  # (B, num_classes)
        return pred


def main(args: Namespace):
    rank = int(os.environ['LOCAL_RANK'])
    IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
    DEVICE = rank
    EPOCHS = args.epochs
    
    if IS_MASTER:
        _, log_filename, best_filename, last_filename = prepare_lineval_dir(
            args.expname, 
            tag=str(args.tag), 
            dataset='imagenet1k', 
            args=vars(args)
        )
    
    # DataLoaders, Models
    train_loader, test_loader, _ = get_imagenet_dataloaders(
        args.batch_size//world_size, args.test_batch_size//world_size,
        args.num_workers, use_ddp=True,
    )
    if args.timm_model is not None:
        if IS_MASTER:
            print(f'Loading {args.timm_model} from timm')
        model = imagenet_model_dict[args.timm_model](pretrained=True)
    else:
        model, _ = load_from_checkpoint(args.expname, tag=args.tag)
    model: ModelBase = model.cuda(DEVICE)
    head = ClsHead(model.embed_dim).cuda(DEVICE)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    head = nn.parallel.DistributedDataParallel(head, device_ids=[rank])
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
    best_top1 = -torch.inf
    train_loss_list, test_loss_list = [], []
    train_top1_list, test_top1_list = [], []
    train_top5_list, test_top5_list = [], []
    for epoch in range(args.epochs):
        
        with tqdm(train_loader, desc=f'TRAIN {epoch+1}', dynamic_ncols=True, disable=not IS_MASTER) as bar:
            total_loss, correct_top1, correct_top5, total = 0, 0, 0, 0
            for input, target, _ in bar:
                with torch.no_grad():
                    x = model.forward(input.cuda(DEVICE))[1]['feats'][-1]
                pred_logit = head(x) # (B, 1000)

                loss = F.cross_entropy(pred_logit, target.to(DEVICE))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                with torch.no_grad():
                    loss = F.cross_entropy(pred_logit, target.to(DEVICE), reduction='none')
                    loss_all = dist_fn.gather(loss)
                    
                    i_top5_all = dist_fn.gather(pred_logit).topk(k=5, dim=1).indices
                    target_all = dist_fn.gather(target.to(DEVICE)).unsqueeze(-1)
                    
                    total_loss += loss_all.sum().cpu().item()
                    correct_top1 += (i_top5_all[:, 0:1] == target_all).sum().cpu().item()
                    correct_top5 += (i_top5_all[:, 0:5] == target_all).sum().cpu().item()
                    total += loss_all.size(0)
                    
                    if IS_MASTER:
                        bar.set_postfix(dict(
                            lr=optimizer.param_groups[0]['lr'],
                            loss=total_loss/total,
                            top1=correct_top1/total*100.0,
                            top5=correct_top5/total*100.0,
                        ))
            train_top1 = correct_top1 / total
            train_top5 = correct_top5 / total
            train_loss = total_loss / total

        with tqdm(test_loader, desc=f' TEST {epoch+1}', dynamic_ncols=True, disable=not IS_MASTER) as bar, torch.no_grad():
            total_loss, correct_top1, correct_top5, total = 0, 0, 0, 0
            for input, target in bar:
                x = model.forward(input.cuda(DEVICE))[1]['feats'][-1]
                pred_logit = head(x) # (B, 1000)
                
                loss = F.cross_entropy(pred_logit, target.to(DEVICE), reduction='none', ignore_index=-1)
                loss_all = dist_fn.gather(loss)
                    
                i_top5_all = dist_fn.gather(pred_logit).topk(k=5, dim=1).indices
                target_all = dist_fn.gather(target.to(DEVICE)).unsqueeze(-1)
                
                total_loss += loss_all.sum().cpu().item()
                correct_top1 += (i_top5_all[:, 0:1] == target_all).sum().cpu().item()
                correct_top5 += (i_top5_all[:, 0:5] == target_all).sum().cpu().item()
                total += loss_all.size(0)
                
                if IS_MASTER:
                    bar.set_postfix(dict(
                        loss=total_loss/total,
                        top1=correct_top1/total*100.0,
                        top5=correct_top5/total*100.0,
                    ))
            test_top1 = correct_top1 / total
            test_top5 = correct_top5 / total
            test_loss = total_loss / total
        
        # Logging
        train_loss_list.append(train_loss)
        train_top1_list.append(train_top1)
        train_top5_list.append(train_top5)
        test_loss_list.append(test_loss)
        test_top1_list.append(test_top1)
        test_top5_list.append(test_top5)
        
        if IS_MASTER:
            with open(log_filename, 'a') as file:
                print(f'- epoch: {epoch+1}', file=file)
                print(f'  train_loss: {train_loss:.4f}', file=file)
                print(f'  train_top1: {train_top1*100.0:.2f}%', file=file)
                print(f'  train_top5: {train_top5*100.0:.2f}%', file=file)
                print(f'  test_loss: {test_loss:.4f}', file=file)
                print(f'  test_top1: {test_top1*100.0:.2f}%', file=file)
                print(f'  test_top5: {test_top5*100.0:.2f}%', file=file)
                print(file=file)
            
            ckpt = dict(
                epoch=epoch+1,
                train_loss=train_loss_list,
                train_top1=train_top1_list,
                train_top5=train_top5_list,
                test_loss=test_loss_list,
                test_top1=test_top1_list,
                test_top5=test_top5_list,
                head={
                    key: val.clone().detach().cpu()
                    for key, val in head.state_dict().items()
                },
            )
            if test_top1 > best_top1:
                best_top1 = test_top1
                torch.save(ckpt, str(best_filename))
            torch.save(ckpt, str(last_filename))


if __name__ == '__main__':
    parser = ArgumentParser('lineval.imagenet1k')
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
