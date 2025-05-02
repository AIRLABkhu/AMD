from argparse import ArgumentParser
from tqdm.auto import tqdm

import torch
from torch import optim
from torch.optim import lr_scheduler

from mdistiller.dataset.imagenet import get_imagenet_dataloaders
from mdistiller.models._base import ModelBase

from tools.lineval.utils import (
    init_parser,
    prepare_lineval_dir,
    load_from_checkpoint,
)

if __name__ == '__main__':
    parser = ArgumentParser('lineval.imagenet')
    init_parser(parser)
    args = parser.parse_args()
    
    DEVICE = args.device
    EPOCHS = args.epochs
    
    log_dir, log_filename, best_filename, last_filename = prepare_lineval_dir(
        args.expname, 
        tag=args.tag, 
        dataset='imagenet', 
        args=vars(args)
    )
    
    # DataLoaders, Models
    train_loader, test_loader, _ = get_imagenet_dataloaders(
        args.batch_size, args.test_batch_size,
        args.num_workers, use_ddp=False,
    )
    model, _ = load_from_checkpoint(args.expname, tag=args.tag, expected_arch='transformer')
    model: ModelBase = model.cuda(DEVICE)
    head = torch.nn.Linear(model.get_head().in_features, 1000).cuda(DEVICE)
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
    best_top1 = -1
    train_loss_list, train_top1_list, test_loss_list, test_top1_list = [], [], [], []
    for epoch in range(args.epochs):
        
        with tqdm(train_loader, desc=f'TRAIN {epoch+1}', dynamic_ncols=True) as bar:
            total_loss, correct, total = 0, 0, 0
            for input, target, _ in bar:
                with torch.no_grad():
                    x = model.forward_stem(input.cuda(DEVICE))
                    x = model.get_layers().forward(x)
                    x = model.forward_pool(x)
                logit = head.forward(x)
                pred = logit.argmax(dim=1).clone().detach().cpu()
                
                loss = torch.nn.functional.cross_entropy(logit, target.cuda(DEVICE))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                batch_size = input.size(0)
                total_loss += loss.clone().detach().cpu().item() * batch_size
                correct += (pred == target).sum().item()
                total += batch_size
                bar.set_postfix(dict(
                    top1=correct/total*100,
                    loss=total_loss/total,
                    lr=optimizer.param_groups[0]['lr'],
                ))
            train_top1 = correct / total * 100
            train_loss = total_loss / total

        with tqdm(test_loader, desc=f' TEST {epoch+1}', dynamic_ncols=True) as bar, torch.no_grad():
            total_loss, correct, total = 0, 0, 0
            for input, target in bar:
                x = model.forward_stem(input.cuda(DEVICE))
                x = model.get_layers().forward(x)
                x = model.forward_pool(x)
                logit = head.forward(x)
                pred = logit.argmax(dim=1).cpu()
                loss = torch.nn.functional.cross_entropy(logit, target.cuda(DEVICE))
                
                batch_size = input.size(0)
                total_loss += loss.cpu().item() * batch_size
                correct += (pred == target).sum().item()
                total += batch_size
                bar.set_postfix(dict(
                    top1=correct/total*100,
                    loss=total_loss/total,
                ))
            test_top1 = correct / total * 100
            test_loss = total_loss / total
        
        # Logging
        train_loss_list.append(train_loss)
        train_top1_list.append(train_top1)
        test_loss_list.append(test_loss)
        test_top1_list.append(test_top1)
        
        with open(log_filename, 'a') as file:
            print(f'- epoch: {epoch+1}', file=file)
            print(f'  train_loss: {train_loss:.4f}', file=file)
            print(f'  train_top1: {train_top1:.4f}', file=file)
            print(f'  test_loss: {test_loss:.4f}', file=file)
            print(f'  test_top1: {test_top1:.4f}', file=file)
            print(file=file)
        
        ckpt = dict(
            epoch=epoch+1,
            train_loss=train_loss_list,
            train_top1=train_top1_list,
            test_loss=test_loss_list,
            test_top1=test_top1_list,
            head={
                key: val.clone().detach().cpu()
                for key, val in head.state_dict().items()
            },
        )
        if test_top1 > best_top1:
            best_top1 = test_top1
            torch.save(ckpt, str(best_filename))
        torch.save(ckpt, str(last_filename))
