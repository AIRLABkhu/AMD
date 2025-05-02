import os
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import sys
import time
from tqdm import tqdm
from ..utils import dist_fn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, distiller):
    IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
    
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    if IS_MASTER:
        pbar = tqdm(range(num_iter), dynamic_ncols=True)

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            
            target_all = dist_fn.gather(target)
            output_all = dist_fn.gather(output)
            loss = criterion(output_all, target_all)
            batch_size = len(target_all)
            
            acc1, acc5 = accuracy(output_all, target_all, topk=(1, 5))
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            # measure elapsed time
            if IS_MASTER:
                msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                    top1=top1, top5=top5
                )
                pbar.set_description(log_msg(msg, "EVAL"))
                pbar.update()
    if IS_MASTER:
        pbar.close()
    return top1.avg, top5.avg, losses.avg


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, bidx, cfg, optimizer):
    match cfg.SOLVER.SCHEDULE.TYPE:
        
        case 'MULTISTEP':
            if bidx == 0:
                steps = np.sum(epoch > np.asarray(cfg.SOLVER.SCHEDULE.MULTISTEP.STAGES))
                if steps > 0:
                    new_lr = cfg.SOLVER.LR * (cfg.SOLVER.SCHEDULE.MULTISTEP.RATE**steps)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr
                    return new_lr
                return cfg.SOLVER.LR
            else:
                return optimizer.param_groups[0]["lr"]
        
        case 'COSINE':
            from math import ceil
            warmup_epochs = cfg.SOLVER.SCHEDULE.COSINE.WARMUP
            num_epoch_batches = int(ceil({
                'imagenet': 1281167,
                'cifar100': 50000,
            }[cfg.DATASET.TYPE] / cfg.SOLVER.BATCH_SIZE))
            num_warmup_batches = num_epoch_batches * warmup_epochs
            
            base_lr = cfg.SOLVER.LR
            global_bidx = (epoch - 1) * num_epoch_batches + bidx
            if global_bidx < num_warmup_batches:
                new_lr = base_lr / num_warmup_batches * (global_bidx + 1)
            else:
                cos_bidx = global_bidx - num_warmup_batches
                num_cos_batches = num_epoch_batches * (cfg.SOLVER.EPOCHS - warmup_epochs)
                last_lr = base_lr * cfg.SOLVER.SCHEDULE.COSINE.RATE
                new_lr = (np.cos(cos_bidx / num_cos_batches * np.pi) + 1.0) * 0.5 * (base_lr - last_lr) + last_lr
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            return new_lr
            
        case _:
            raise NotImplementedError(cfg.SOLVER.SCHEDULE.TYPE)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)
