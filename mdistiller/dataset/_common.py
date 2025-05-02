from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, use_ddp: bool):
    if use_ddp:
        sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
        loader = DataLoader(dataset, sampler=sampler, pin_memory=True, batch_size=batch_size, num_workers=num_workers)
    else:
        loader = DataLoader(dataset, shuffle=shuffle, pin_memory=True, batch_size=batch_size, num_workers=num_workers)
    return loader
