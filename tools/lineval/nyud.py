from argparse import ArgumentParser
from tqdm.auto import tqdm
import numpy as np

import einops

import torch
from torch import optim
from torch.optim import lr_scheduler
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


if __name__ == '__main__':
    parser = ArgumentParser('lineval.nyud')
    init_parser(parser, defaults=dict(epochs=1000))
    args = parser.parse_args()
    
    DEVICE = args.device
    EPOCHS = args.epochs
    
    log_dir, log_filename, best_filename, last_filename = prepare_lineval_dir(
        args.expname, 
        tag=args.tag, 
        dataset='nyud', 
        args=vars(args)
    )
    
    # DataLoaders, Models
    train_loader, test_loader, _ = get_nyud_dataloaders(
        args.batch_size, args.test_batch_size,
        args.num_workers, use_ddp=False,
    )
    if args.timm_model is not None:
        print(f"Loading {args.timm_model} from timm")
        model = imagenet_model_dict[args.timm_model](pretrained=True)
    else:
        model, _ = load_from_checkpoint(args.expname, tag=args.tag)
    model: ModelBase = model.cuda(DEVICE)
    head = torch.nn.Linear(model.embed_dim, 256).cuda(DEVICE)
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
    
    # Utility
    def crop_resize(x: torch.Tensor, size: int, random_crop: bool=False):
        shorter = min(x.shape[-2:])
        x_cropped = center_crop(x, shorter)
        if random_crop:
            crop_size = int(shorter * 0.8)
            low = np.random.randint(0, shorter - crop_size - 1, size=(2,))
            high = low + crop_size
            x_cropped = x[..., low[0]:high[0], low[1]:high[1]]
        x_resized = resize(x_cropped, (size, size))
        return x_resized

    def restore_depth_map(x: torch.Tensor, size: tuple[int, int]):
        return resize(x, size)

    def patches_to_depth(x: torch.Tensor):
        _, num_patches, embed_dim = x.shape
        patch_rows = int(num_patches**0.5)
        pixel_rows = int(embed_dim**0.5)
        return einops.rearrange(
            x[:, -patch_rows*patch_rows:],
            'b (ph pw) (h w) -> b (ph h) (pw w)',
            ph=patch_rows, pw=patch_rows,
            h=pixel_rows, w=pixel_rows,
        )
    
    # Training Loop
    best_rmse = torch.inf
    train_loss_list, train_rmse_list, test_loss_list, test_rmse_list = [], [], [], []
    for epoch in range(args.epochs):
        
        with tqdm(train_loader, desc=f'TRAIN {epoch+1}', dynamic_ncols=True) as bar:
            total_loss, total_rmse, total = 0, 0, 0
            for input, target in bar:
                input = crop_resize(input, size=224, random_crop=True)
                with torch.no_grad():
                    x = model.forward_stem(input.cuda(DEVICE))
                    x = model.get_layers().forward(x)
                pred_patches = head.forward(x)
                pred = patches_to_depth(pred_patches)
                pred = restore_depth_map(pred, target.shape[-2:])
                
                loss = torch.nn.functional.mse_loss(pred, target.to(DEVICE))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                batch_size = input.size(0)
                total_loss += loss.clone().detach().cpu().item() * batch_size
                total_rmse += torch.square(pred - target.to(DEVICE)).flatten(1) \
                    .mean(dim=1).sqrt().sum().clone().detach().cpu().item()
                total += batch_size
                bar.set_postfix(dict(
                    loss=total_loss/total,
                    rmse=total_rmse/total,
                    lr=optimizer.param_groups[0]['lr'],
                ))
            train_rmse = total_rmse / total
            train_loss = total_loss / total

        with tqdm(test_loader, desc=f' TEST {epoch+1}', dynamic_ncols=True) as bar, torch.no_grad():
            total_loss, total = 0, 0
            for input, target in bar:
                input = crop_resize(input, size=224, random_crop=False)
                x = model.forward_stem(input.cuda(DEVICE))
                x = model.get_layers().forward(x)
                pred_patches = head.forward(x)
                pred = patches_to_depth(pred_patches)
                pred = restore_depth_map(pred, target.shape[-2:])
                loss = torch.nn.functional.mse_loss(pred, target.to(DEVICE))
                
                batch_size = input.size(0)
                total_loss += loss.cpu().item() * batch_size
                total_rmse += torch.square(pred - target.to(DEVICE)).flatten(1) \
                    .mean(dim=1).sqrt().sum().cpu().item()
                total += batch_size
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
        if test_rmse > best_rmse:
            best_rmse = test_rmse
            torch.save(ckpt, str(best_filename))
        torch.save(ckpt, str(last_filename))