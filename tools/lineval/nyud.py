import os
from argparse import ArgumentParser, Namespace
from tqdm.auto import tqdm
from typing import Literal
import numpy as np

import einops

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch import distributed as dist
from torchvision.transforms.functional import (
    resize, center_crop
)

from mdistiller.dataset.nyud_v2 import get_nyud_dataloaders
from mdistiller.models._base import ModelBase
from mdistiller.models import imagenet_model_dict

from tools.lineval.utils import (
    init_parser,
    prepare_lineval_dir,
    load_from_checkpoint,
)
from mdistiller.utils import dist_fn


# Utility
def crop_resize(*x: tuple[torch.Tensor, ...], size: int, random_crop: bool=False):
    x_repr = x[0]
    shorter = min(x_repr.shape[-2:])
    
    x_cropped = [center_crop(x_, shorter) for x_ in x]
    if random_crop:
        crop_size = int(shorter * 0.8)
        low = np.random.randint(0, shorter - crop_size - 1, size=(2,))
        high = low + crop_size
        x_cropped = [
            x_[..., low[0]:high[0], low[1]:high[1]]
            for x_ in x_cropped
        ]
    x_resized = [resize(x_, (size, size)) for x_ in x_cropped]
    return x_resized

def patch_to_depth(
    x: torch.Tensor, 
    size: tuple[int, int], 
    head: nn.Linear, 
):
    '''
    x: torch.Tensor.shape(B, (h, w), C)
    ret: torch.Tensor.shape(B, H, W, D)
    '''
    num_rows = int(x.size(1)**0.5)
    num_patches = num_rows * num_rows
    x = x[..., -num_patches:, :]
    x = einops.rearrange(x, 'B (h w) C -> B C h w', h=num_rows)
    x_resized = resize(x, size=size)
    x_resized = x_resized.permute(0, 2, 3, 1)  # B H W C
    x_depth = head.forward(x_resized)  # B H W D
    return x_depth

def reform_depth(
    x: torch.Tensor,
    kind: Literal['logit', 'depth']='logit',
):
    '''
    x: torch.Tensor.shape(B, H, W, D)
    ret: 
        logit: torch.Tensor.shape(B, H, W, D)
        depth: torch.Tensor.shape(B, H, W)
    '''
    match kind:
        case 'logit':
            x = x.softmax(dim=-1)
            x = x.permute(0, 3, 1, 2)
        case 'depth':
            resolution = x.size(-1)
            depth_list = torch.linspace(0, 1, resolution, dtype=x.dtype, device=x.device)
            depth_list = depth_list.reshape(1, 1, 1, resolution)
            x_hard = nn.functional.one_hot(
                x.argmax(dim=-1),
                num_classes=resolution,
            )
            x = x_hard - x.detach() + x
            x = x.sum(dim=-1)
        case _:
            raise NotImplementedError(kind)
    return x

    
def main(args: Namespace):
    rank = int(os.environ['LOCAL_RANK'])
    IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
    DEVICE = rank
    EPOCHS = args.epochs
    
    if IS_MASTER:
        _, log_filename, best_filename, last_filename = prepare_lineval_dir(
            args.expname, 
            tag=str(args.tag), 
            dataset='nyud', 
            args=vars(args)
        )
    
    # DataLoaders, Models
    train_loader, test_loader, _ = get_nyud_dataloaders(
        args.batch_size//world_size, args.test_batch_size//world_size,
        args.num_workers, use_ddp=True,
    )
    if args.timm_model is not None:
        print(f"Loading {args.timm_model} from timm")
        model = imagenet_model_dict[args.timm_model](pretrained=True)
    else:
        model, _ = load_from_checkpoint(args.expname, tag=args.tag)
    model: ModelBase = model.cuda(DEVICE)
    depth_resolution = 256
    head = torch.nn.Linear(model.embed_dim, depth_resolution).cuda(DEVICE)
    
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
    best_rmse = torch.inf
    train_loss_list, train_rmse_list, test_loss_list, test_rmse_list = [], [], [], []
    for epoch in range(args.epochs):
        
        with tqdm(train_loader, desc=f'TRAIN {epoch+1}', dynamic_ncols=True, disable=not IS_MASTER) as bar:
            total_loss, total_rmse, total = 0, 0, 0
            for input, target in bar:
                input, = crop_resize(input, size=224, random_crop=True)
                with torch.no_grad():
                    x = model.forward(input.cuda(DEVICE))[1]['feats'][-1]
                pred = patch_to_depth(x, size=target.shape[-2:], head=head)
                pred_logit = reform_depth(pred, kind='logit')
                
                target_binned = (target * (depth_resolution - 1)).round().long()
                loss = torch.nn.functional.cross_entropy(pred_logit, target_binned.to(DEVICE), reduction='mean')
                loss.backward()
                if IS_MASTER: print(head.module.weight.grad.norm(), f'{loss:.8f}')
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                # if IS_MASTER: print(head.module.weight.grad.norm(), f'{loss:.8f}')
                
                with torch.no_grad():
                    pred_depth = reform_depth(pred, kind='depth')
                    loss = torch.nn.functional.cross_entropy(pred_logit, target_binned.to(DEVICE), reduction='none')
                    loss_all = dist_fn.gather(loss)
                    local_rmse = torch.square(pred_depth - target.to(DEVICE)).flatten(1).mean(dim=1).sqrt()
                    rmse_all = dist_fn.gather(local_rmse)
                    
                    total_loss += loss_all.mean().cpu().item()
                    total_rmse += rmse_all.sum().cpu().item()
                    total += loss_all.size(0)
                    
                    if IS_MASTER:
                        bar.set_postfix(dict(
                            lr=optimizer.param_groups[0]['lr'],
                            loss=total_loss/total,
                            rmse=total_rmse/total,
                        ))
            train_rmse = total_rmse / total
            train_loss = total_loss / total

        with tqdm(test_loader, desc=f' TEST {epoch+1}', dynamic_ncols=True, disable=not IS_MASTER) as bar, torch.no_grad():
            total_loss, total_rmse, total = 0, 0, 0
            for input, target in bar:
                input, target = crop_resize(input, target, size=224, random_crop=False)
                x = model.forward(input.cuda(DEVICE))[1]['feats'][-1]
                pred = patch_to_depth(x, size=target.shape[-2:], head=head)
                pred_logit = reform_depth(pred, kind='logit')
                pred_depth = reform_depth(pred, kind='depth')
                
                target_binned = (target * (depth_resolution - 1)).round().long()
                loss = torch.nn.functional.cross_entropy(pred_logit, target_binned.to(DEVICE), reduction='none')
                
                loss_all = dist_fn.gather(loss)
                local_rmse = torch.square(pred_depth - target.to(DEVICE)).flatten(1).mean(dim=1).sqrt()
                rmse_all = dist_fn.gather(local_rmse)
                
                total_loss += loss_all.mean().cpu().item()
                total_rmse += rmse_all.sum().cpu().item()
                total += loss_all.size(0)
                
                if IS_MASTER:
                    bar.set_postfix(dict(
                        loss=total_loss/total,
                        rmse=total_rmse/total,
                    ))
            test_rmse = total_rmse / total
            test_loss = total_loss / total
        
        # Logging
        train_loss_list.append(train_loss)
        train_rmse_list.append(train_rmse)
        test_loss_list.append(test_loss)
        test_rmse_list.append(test_rmse)
        
        if IS_MASTER:
            with open(log_filename, 'a') as file:
                print(f'- epoch: {epoch+1}', file=file)
                print(f'  train_loss: {train_loss:.4f}', file=file)
                print(f'  train_rmse: {train_rmse:.4f}', file=file)
                print(f'  test_loss: {test_loss:.4f}', file=file)
                print(f'  test_rmse: {test_rmse:.4f}', file=file)
                print(file=file)
            
            ckpt = dict(
                epoch=epoch+1,
                train_loss=train_loss_list,
                train_rmse=train_rmse_list,
                test_loss=test_loss_list,
                test_rmse=test_rmse_list,
                head={
                    key: val.clone().detach().cpu()
                    for key, val in head.state_dict().items()
                },
            )
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                torch.save(ckpt, str(best_filename))
            torch.save(ckpt, str(last_filename))


if __name__ == '__main__':
    parser = ArgumentParser('lineval.nyud')
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
