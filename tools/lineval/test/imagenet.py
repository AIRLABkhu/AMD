from argparse import ArgumentParser, Namespace
from tqdm.auto import tqdm

import torch
from mdistiller.dataset.imagenet import get_imagenet_dataloaders
from mdistiller.models._base import ModelBase
from mdistiller.models import imagenet_model_dict

from tools.lineval.utils import (
    init_parser,
    load_from_checkpoint,
    load_head_checkpoint,
)
from tools.lineval.imagenet import (
    ClsHead,
)


def compute_cls_metrics(pred, gt):
    """
    pred: torch.Tensor of shape (B, C), float type
    gt: torch.Tensor of shape (B), long type
    Returns: dict with average metrics over batch
    """
    top5_indices = pred.topk(k=5, dim=1).indices
    targets = gt.reshape(-1, 1)
    
    batch_size = len(pred)
    top1 = (top5_indices[:, 0:1] == targets).sum() / batch_size
    top5 = (top5_indices[:, 0:5] == targets).sum() / batch_size

    return {
        'top1': top1.cpu().item(),
        'top5': top5.cpu().item(),
    }


def main(args: Namespace):
    DEVICE = args.device
    
    # DataLoaders, Models
    _, test_loader, _ = get_imagenet_dataloaders(
        1, args.batch_size,
        args.num_workers, use_ddp=False,
    )
    if args.timm_model is not None:
        print(f"Loading {args.timm_model} from timm")
        model = imagenet_model_dict[args.timm_model](pretrained=True)
    else:
        model, _ = load_from_checkpoint(args.expname, tag=args.tag)
    model: ModelBase = model.cuda(DEVICE)
    head = ClsHead(model.embed_dim).cuda(DEVICE)
    
    head_state_dict = load_head_checkpoint(
        args.expname, tag=args.tag, 
        when=args.lineval_when, dataset='imagenet1k',
        lineval_tag='best'
    )['head']
    head_state_dict = {
        key.replace('module.', ''): val
        for key, val in head_state_dict.items()
    }
    print(head.load_state_dict(head_state_dict))
    
    # Training Loop
    with tqdm(test_loader, desc=f' TEST', dynamic_ncols=True) as bar, torch.no_grad():
        total_loss, preds_all, targets_all = 0, [], []
        for input, target in bar:
            target = target.long()
            x = model.forward(input.cuda(DEVICE))[1]['feats'][-1]
            pred_logit = head.forward(x) # (B, 1000)
            loss = torch.nn.functional.cross_entropy(pred_logit, target.to(DEVICE))
            
            batch_size = input.size(0)
            total_loss += loss.cpu().item() * batch_size
            preds_all.append(pred_logit.cpu())
            targets_all.append(target.cpu())
        
    preds = torch.cat(preds_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    mean_loss = total_loss / len(preds)
    
    metrics = compute_cls_metrics(preds, targets)
    metrics['loss'] = mean_loss
    max_key_len = max(map(len, metrics.keys()))
    for key, val in metrics.items():
        print(f'{key:<{max_key_len}}: {val:.4f}')


if __name__ == '__main__':
    parser = ArgumentParser('lineval.test.imagenet1k')
    init_parser(parser, defaults=dict(batch_size=128))
    parser.add_argument('--lineval-when', '-lw', type=str, default=None)
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        pass
