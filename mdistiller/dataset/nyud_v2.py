import os 
from typing import Literal
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ._common import make_loader


DATAROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/nyud')
MEAN = (0.485, 0.425, 0.385)
STD = (0.229, 0.224, 0.225)


def denormalize(x: torch.Tensor):
    tensor_metadata = dict(dtype=x.dtype, device=x.device)
    channel_at = np.nonzero(np.array(x.shape) == 3)[0][0]
    match x.ndim:
        case 3:
            stat_shape = [1] * 3
            stat_shape[channel_at] = 3
        case 4:
            stat_shape = [1] * 4
            stat_shape[channel_at+1] = 3
        case _:
            raise RuntimeError
    mean = torch.tensor(MEAN, **tensor_metadata).reshape(stat_shape)
    std = torch.tensor(STD, **tensor_metadata).reshape(stat_shape)
    return x * std + mean

def get_nyud_train_transform(mean=MEAN, std=STD, img_size: int=224):
    resizing_size = int(img_size * (256 / 224))
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.1, 
            hue=0.05, 
            p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], additional_targets={'depth': 'image'})


def get_nyud_test_transform(mean=MEAN, std=STD, img_size: int=224):
    resizing_size = int(img_size * (256 / 224))
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], additional_targets={'depth': 'image'})


class NYUdV2(Dataset):
    def __init__(
        self,
        dataroot: str=DATAROOT,
        split: Literal['train', 'test']='train',
        transform=None,
    ):
        self.file = h5py.File(os.path.join(dataroot, 'nyud_v2.hdf5'), 'r')
        self.dataset = self.file.get(split)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset['images'])
    
    def __getitem__(self, index: int, return_masks: bool=False):
        image = self.dataset['images'][index]
        depth = self.dataset['depths'][index]
        
        depth = np.clip(depth, 0, 10) / 10.0
        
        output = self.transform(image=image, depth=depth)
        image: np.ndarray = output['image']
        depth: np.ndarray = output['depth'][0]
        
        if return_masks:
            return image, depth, depth > 0
        else:
            return image, depth


def get_nyud_val_loader(val_batch_size, use_ddp, mean=MEAN, std=STD, img_size: int=224):
    test_transform = get_nyud_test_transform(mean, std, img_size=img_size)
    test_set = NYUdV2(split='test', transform=test_transform)
    test_loader = make_loader(test_set, val_batch_size, num_workers=16, shuffle=False, use_ddp=use_ddp)
    return test_loader

def get_nyud_dataloaders(batch_size, val_batch_size, num_workers, use_ddp,
    mean=MEAN, std=STD, img_size: int=224):
    train_transform = get_nyud_train_transform(mean, std, img_size=img_size)
    train_set = NYUdV2(split='train', transform=train_transform)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_nyud_val_loader(val_batch_size, use_ddp, mean, std, img_size=img_size)
    return train_loader, test_loader, num_data
