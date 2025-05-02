import os
from datetime import datetime
from argparse import ArgumentParser
from typing import Literal
from pathlib import Path
from yacs.config import CfgNode as CN
import torch
from mdistiller.models.imagenet import imagenet_model_dict

def get_config(
    exp_name: str,
    ckpt_tag: Literal['latest', 'best']|int|None=None,
):
    exp_root = os.path.join('output', exp_name)
    with open(os.path.join(exp_root, 'code', '_cfg.yaml'), 'r') as file:
        cfg = CN.load_cfg(file)
    if ckpt_tag is None:
        return cfg
    else:
        ckpt = torch.load(os.path.join(exp_root, f'student_{ckpt_tag}'), map_location='cpu', weights_only=False)
        return cfg, ckpt

def prepare_lineval_dir(
    exp_name: str,
    tag: Literal['latest', 'best']|int='latest',
    dataset: str='imagenet',
    args: dict|None=None,
):
    lineval_dir = Path('output').joinpath(exp_name, 'lineval')
    nowstr = datetime.now().strftime('_%y%m%d_%H%M%S')
    log_dir = lineval_dir.joinpath(tag, dataset + nowstr)
    log_dir.mkdir(parents=True)
    
    if args is not None:
        cfg_filename = log_dir.joinpath('_cfg.yaml')
        with open(cfg_filename, 'w') as file:
            for key, val in args.items():
                print(f'{key}: {val}', file=file)
    
    log_filename = log_dir.joinpath('log.yaml')
    best_filename = log_dir.joinpath('best.pt')
    last_filename = log_dir.joinpath('last.pt')
    return log_dir, log_filename, best_filename, last_filename

def load_from_checkpoint(
    exp_name: str,
    tag: Literal['latest', 'best']|int='latest',
    expected_arch: Literal['cnn', 'transformer']|None=None,
) -> torch.nn.Module:
    '''
    :Example:
        ```
        checkpoint = load_from_checkpoint(
            exp_name='imagenet_baselines/fitnet,res34,res18',
            tag='latest',
        )
        model, load_state_dict_result = checkpoint
        ```
    '''
    cfg, ckpt = get_config(exp_name, ckpt_tag=tag)
    model: torch.nn.Module = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
    if (expected_arch is not None) and (model.get_arch() != expected_arch):
        raise ValueError(f'Expected {expected_arch}, but this checkpoint requires {model.get_arch()}.')
    result = model.load_state_dict(ckpt['model'], strict=False)
    return model, result


if __name__ == '__main__':
    model, result = load_from_checkpoint('imagenet_baselines/fitnet,res34,res18', tag='latest')
    print(result)

def T_CHECKPOINT_TAG(tag: str):
    if tag in {'latest', 'best'}:
        return tag
    elif tag.isdigit():
        return int(tag)
    else:
        raise ValueError

def init_parser(
    parser: ArgumentParser, defaults: dict={},
) -> ArgumentParser:
    # experiment
    parser.add_argument('expname', type=str)
    parser.add_argument('--tag', '-t', type=T_CHECKPOINT_TAG, default='latest')
    parser.add_argument('--device', '-d', type=int, default=0)
    # dataloader
    parser.add_argument('--batch-size', '-bs', type=int, default=512)
    parser.add_argument('--test-batch-size', '-tbs', type=int, default=512)
    parser.add_argument('--num-workers', '-nw', type=int, default=8)
    # optimizer
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1.0E-6)

    # for load pretrained model from timm
    parser.add_argument('--timm-model', type=str, default=None)
    
    parser.set_defaults(**defaults)
    return parser
