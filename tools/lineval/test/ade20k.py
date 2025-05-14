from argparse import ArgumentParser, Namespace
from tqdm.auto import tqdm

import torch
from mdistiller.dataset.ade20k import get_ade20k_dataloaders
from mdistiller.models._base import ModelBase
from mdistiller.models import imagenet_model_dict

from tools.lineval.utils import (
    init_parser,
    load_from_checkpoint,
    load_head_checkpoint,
)
from tools.lineval.ade20k import (
    SemSegHead,
    MulticlassJaccardIndex,
)


def compute_semseg_metrics(pred, gt):
    """
    pred: torch.Tensor of shape (B, C, H, W), float type
    gt: torch.Tensor of shape (B, H, W), long type
    mask: optional torch.BoolTensor of shape (B, H, W)
    Returns: dict with average metrics over batch
    """
    pred = pred.float()
    gt = gt.long()
    miou = MulticlassJaccardIndex(150, ignore_index=-1)

    return {
        'mIoU': miou(pred, gt).item()*100.0,
    }


def main(args: Namespace):
    DEVICE = args.device
    
    # DataLoaders, Models
    _, test_loader, _ = get_ade20k_dataloaders(
        1, args.batch_size,
        args.num_workers, use_ddp=False,
    )
    if args.timm_model is not None:
        print(f"Loading {args.timm_model} from timm")
        model = imagenet_model_dict[args.timm_model](pretrained=True)
    else:
        model, _ = load_from_checkpoint(args.expname, tag=args.tag)
    model: ModelBase = model.cuda(DEVICE)
    head = SemSegHead(model.embed_dim, upsample_factor=16).cuda(DEVICE)
    
    head_state_dict = load_head_checkpoint(
        args.expname, tag=args.tag, 
        when=args.lineval_when, dataset='ade20k',
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
            pred_logit = head.forward(x) # (B, 150, H, W)
            loss = torch.nn.functional.cross_entropy(pred_logit, target.to(DEVICE), ignore_index=-1)
            
            batch_size = input.size(0)
            total_loss += loss.cpu().item() * batch_size
            preds_all.append(pred_logit.cpu())
            targets_all.append(target.cpu())
        
    preds = torch.cat(preds_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    mean_loss = total_loss / len(preds)
    
    metrics = compute_semseg_metrics(preds, targets)
    metrics['loss'] = mean_loss
    max_key_len = max(map(len, metrics.keys()))
    for key, val in metrics.items():
        print(f'{key:<{max_key_len}}: {val:.4f}')


if __name__ == '__main__':
    parser = ArgumentParser('lineval.test.semseg')
    init_parser(parser, defaults=dict(batch_size=128))
    parser.add_argument('--lineval-when', '-lw', type=str, default=None)
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        pass
