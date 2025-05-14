from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


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


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, use_ddp: bool):
    if use_ddp:
        sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
        loader = DataLoader(dataset, sampler=sampler, pin_memory=True, batch_size=batch_size, num_workers=num_workers)
    else:
        loader = DataLoader(dataset, shuffle=shuffle, pin_memory=True, batch_size=batch_size, num_workers=num_workers)
    return loader
