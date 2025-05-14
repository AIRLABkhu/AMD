import os 
from typing import Literal
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ._common import make_loader, SafeColorJitter

# from 
# https://github.com/yassouali/pytorch-segmentation/blob/master/dataloaders/ade20k.py

DATAROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/ade20k')
MEAN = (0.48897059, 0.46548275, 0.4294)
STD = (0.22861765, 0.22948039, 0.24054667)

COLORMAP = np.array([
    [120, 120, 120], [180, 120, 120], [  6, 230, 230], [ 80,  50,  50], [  4, 200,   3],
    [120, 120,  80], [140, 140, 140], [204,   5, 255], [230, 230, 230], [  4, 250,   7],
    [224,   5, 255], [235, 255,   7], [150,   5,  61], [120, 120,  70], [  8, 255,  51],
    [255,   6,  82], [143, 255, 140], [204, 255,   4], [255,  51,   7], [204,  70,   3],
    [  0, 102, 200], [ 61, 230, 250], [255,   6,  51], [ 11, 102, 255], [255,   7,  71],
    [255,   9, 224], [  9,   7, 230], [220, 220, 220], [255,   9,  92], [112,   9, 255],
    [  8, 106,  10], [196, 255, 255], [  7, 255, 224], [255, 184,   6], [ 10, 255,  71],
    [255,  41,  10], [  7, 255, 255], [224, 255,   8], [102,   8, 255], [255,  61,  10],
    [255, 194,   7], [255, 122,   8], [  0, 255,  20], [255,   8,  41], [255,   5, 153],
    [  6,  51, 255], [235,  12, 255], [160, 150,  20], [  0, 163, 255], [140, 140, 140],
    [250,  10,  15], [ 20, 255,   0], [ 31, 255,   0], [255,  31,   0], [255, 224,   0],
    [153, 255,   0], [  0,  92, 255], [  0, 255,  92], [184,   0, 255], [255,   0, 184],
    [  0, 184, 255], [  0, 214, 255], [255,   0,  92], [  0, 255, 184], [  0,  31, 255],
    [255,  31,   0], [255,  15, 153], [  0,  40, 255], [  0, 255, 204], [ 41,   0, 255],
    [  0, 146, 255], [255, 208,   0], [255, 255,  41], [  0, 255, 204], [  0, 255, 153],
    [255,  92,   0], [255, 153,   0], [255, 204,   0], [  0, 255, 255], [  0, 153, 255],
    [  0, 255, 102], [255, 255,   0], [153,   0, 255], [255,   0, 102], [255,   0, 255],
    [  0, 255,  20], [255, 204,  41], [  0, 255, 153], [  0, 255,   0], [255,  92, 153],
    [204,   0, 255], [255,  61,  92], [255, 153, 153], [  0,  92, 255], [255, 153,  92],
    [  0,  20, 255], [153, 255, 204], [  0,  92, 153], [  0, 153, 204], [153, 204, 255],
    [102, 255, 255], [255, 255, 204], [204, 255, 204], [255, 204, 255], [204, 204, 255],
    [255, 255, 153], [204, 255, 255], [255, 204, 204], [204, 204, 204], [102, 102, 102],
    [255, 153, 204], [255, 102, 204], [204, 153, 255], [204, 255, 153], [153, 204, 153],
    [204, 153, 153], [255, 102, 102], [153, 255, 255], [255, 255, 102], [153, 153, 204],
    [102, 204, 255], [255, 102, 153], [204, 102, 255], [255, 204, 102], [102, 204, 153],
    [153, 102, 255], [102, 153, 204], [255, 255, 255], [  0,   0,   0], [  0,   0,  70],
    [  0,  70,  70], [ 70,   0,  70], [ 70,  70,   0], [ 70,   0,   0], [  0,  70,   0],
    [210,  70,  70], [210, 210, 210], [ 70, 210, 210], [210,  70, 210], [210, 210,  70],
    [  0, 128, 128], [128,   0, 128], [128, 128,   0], [128,   0,   0], [  0, 128,   0],
], dtype=np.uint8)


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


class ADE20k(Dataset):
    def __init__(
        self,
        dataroot: str=DATAROOT,
        split: Literal['train', 'test']='train',
        transform=None,
    ):
        split_name = {
            'train': 'training',
            'test': 'validation',
        }[split]
        self.image_root = os.path.join(dataroot, 'images', split_name)
        self.label_root = os.path.join(dataroot, 'annotations', split_name)
        self.filenames = sorted(os.path.splitext(fn)[0] for fn in os.listdir(self.image_root))
        
        with open(os.path.join(dataroot, 'objectInfo150.txt'), 'r') as file:
            lines = file.readlines()
        self.classes = [
            line.split('\t')[-1].strip()
            for line in lines
        ]

        self.num_classes = 150
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index: int):
        image_path = os.path.join(self.image_root, self.filenames[index] + '.jpg')
        label_path = os.path.join(self.label_root, self.filenames[index] + '.png')

        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = (np.asarray(Image.open(label_path), dtype=np.int32) - 1)  # from -1 to 149
        
        output = self.transform(image=image, label=label)
        image: np.ndarray = output['image']
        label: np.ndarray = output['label']
        
        match label:
            case torch.Tensor():
                label = label.long()
            case np.ndarray():
                label = label.astype(np.int64)
        return image, label


def get_ade20k_val_loader(val_batch_size, use_ddp, mean=MEAN, std=STD, img_size: int=224):
    test_transform = get_ade20k_test_transform(mean, std, img_size=img_size)
    test_set = ADE20k(split='test', transform=test_transform)
    test_loader = make_loader(test_set, val_batch_size, num_workers=16, shuffle=False, use_ddp=use_ddp)
    return test_loader

def get_ade20k_dataloaders(batch_size, val_batch_size, num_workers, use_ddp,
    mean=MEAN, std=STD, img_size: int=224):
    train_transform = get_ade20k_train_transform(mean, std, img_size=img_size)
    train_set = ADE20k(split='train', transform=train_transform)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_ade20k_val_loader(val_batch_size, use_ddp, mean, std, img_size=img_size)
    return train_loader, test_loader, num_data
