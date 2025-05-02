import os 
from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ._common import make_loader

# from 
# https://github.com/yassouali/pytorch-segmentation/blob/master/dataloaders/ade20k.py

DATAROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/ade20k')
MEAN = (0.48897059, 0.46548275, 0.4294)
STD = (0.22861765, 0.22948039, 0.24054667)


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
    ], additional_targets={'label': 'image'})


def get_ade20k_test_transform(mean=MEAN, std=STD, img_size: int=224):
    resizing_size = int(img_size * (256 / 224))
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], additional_targets={'label': 'image'})


class ADE20k(Dataset):
    def __init__(
        self,
        dataroot: str=DATAROOT,
        split: Literal['train', 'test']='train',
        transform=None,
    ):
        with open('{}/{}'.format(dataroot, 'index_ade20k.pkl'), 'rb') as f:
            self.index_ade20k = pkl.load(f)

        split_keyword = 'training' if split == 'train' else 'validation'
        valid_indices = [i for i, folder in enumerate(self.index_ade20k['folder']) if split_keyword in folder]
        self.image_filename_list = [
            os.path.join(dataroot, self.index_ade20k['folder'][i], self.index_ade20k['filename'][i])
            for i in valid_indices]

        self.num_classes = 150
        self.transform = transform
    
    def __len__(self):
        return len(self.image_filename_list)
    
    def __getitem__(self, index: int, return_masks: bool=False):
        image_path = self.image_filename_list[index]
        label_path = self.image_filename_list[index].replace('.jpg', '_seg.png'))

        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32) - 1  # from -1 to 149
        
        output = self.transform(image=image, label=label)
        image: np.ndarray = output['image']
        label: np.ndarray = output['label']
        
        return image, label


def get_ade20k_val_loader(val_batch_size, use_ddp, mean=MEAN, std=STD, img_size: int=224):
    test_transform = get_ade20k_test_transform(mean, std, img_size=img_size)
    test_set = ADE20k(split='test', transform=test_transform)
    test_loader = make_loader(test_set, val_batch_size, num_workers=16, shuffle=False, use_ddp=use_ddp)
    return test_loader

def get_nyud_dataloaders(batch_size, val_batch_size, num_workers, use_ddp,
    mean=MEAN, std=STD, img_size: int=224):
    train_transform = get_ade20k_train_transform(mean, std, img_size=img_size)
    train_set = ADE20k(split='train', transform=train_transform)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_ade20k_val_loader(val_batch_size, use_ddp, mean, std, img_size=img_size)
    return train_loader, test_loader, num_data
