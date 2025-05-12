import os 
from typing import Literal
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ._common import make_loader


DATAROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/ade-20k')
MEAN = (0.48897059, 0.46548275, 0.4294)
STD = (0.22861765, 0.22948039, 0.24054667)

# ADE20K colormap for 150 classes
COLORMAP = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255]
], dtype=np.uint8)


class SafeColorJitter(ImageOnlyTransform):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.jitter = A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1.0 
        )

    def apply(self, img, **params):
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        result = self.jitter(image=img)['image']
        return result


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

def get_ade20k_train_transform(mean=MEAN, std=STD, img_size: int=224):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        SafeColorJitter(
            brightness=(0.8, 1.2), 
            contrast=(0.8, 1.2), 
            saturation=(0.9, 1.1), 
            hue=(-0.05, 0.05), 
            p=0.7),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], additional_targets={'label': 'mask'})

def get_ade20k_test_transform(mean=MEAN, std=STD, img_size: int=224):
    return A.Compose([
        A.SmallestMaxSize(max_size=img_size),
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], additional_targets={'label': 'mask'})


class ADE20K(Dataset):
    def __init__(
        self,
        dataroot: str=DATAROOT,
        split: Literal['train', 'test']='train',
        transform=None,
    ):
        split_dir = {
            'train': 'training',
            'test': 'validation',
        }[split]
        self.image_dir = os.path.join(dataroot, 'images', split_dir)
        self.annot_dir = os.path.join(dataroot, 'annotations', split_dir)
        
        image_fnames = set(map(lambda x: x.replace('.jpg', ''), os.listdir(self.image_dir)))
        annot_fnames = set(map(lambda x: x.replace('.png', ''), os.listdir(self.annot_dir)))
        self.filenames = sorted(set.intersection(
            image_fnames, annot_fnames
        ))
        
        with open(os.path.join(dataroot, 'objectInfo150.txt'), 'r') as file:
            lines = file.readlines()[1:]
        self.classes = list(map(lambda x: x.strip().split('\t')[-1], lines))

        self.num_classes = 150
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index: int):
        image_fname = f'{self.image_dir}{os.sep}{self.filenames[index]}.jpg'
        annot_fname = f'{self.annot_dir}{os.sep}{self.filenames[index]}.png'

        image = np.asarray(Image.open(image_fname).convert('RGB')).astype(np.float32)
        label = np.asarray(Image.open(annot_fname), dtype=np.int32) - 1  # from -1 to 149
        
        output = self.transform(image=image, label=label)
        image: np.ndarray = output['image']
        label: np.ndarray = output['label']
        return image, label


def get_ade_20k_val_loader(val_batch_size, use_ddp, mean=MEAN, std=STD, img_size: int=224):
    test_transform = get_ade20k_test_transform(mean, std, img_size=img_size)
    test_set = ADE20K(split='test', transform=test_transform)
    test_loader = make_loader(test_set, val_batch_size, num_workers=16, shuffle=False, use_ddp=use_ddp)
    return test_loader

def get_ade_20k_dataloaders(batch_size, val_batch_size, num_workers, use_ddp,
    mean=MEAN, std=STD, img_size: int=224):
    train_transform = get_ade20k_train_transform(mean, std, img_size=img_size)
    train_set = ADE20K(split='train', transform=train_transform)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_ade_20k_val_loader(val_batch_size, use_ddp, mean, std, img_size=img_size)
    return train_loader, test_loader, num_data
