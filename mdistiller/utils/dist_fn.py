import os
import torch
import torch.distributed as dist


def is_initialized() -> bool:
    return dist.is_initialized()


def broadcast(x: torch.Tensor, src: int=0) -> torch.Tensor:
    dist.broadcast(x, src=src)
    return x


def scatter(x: torch.Tensor|None=None, src: int=0) -> torch.Tensor:
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    assert len(x) % world_size == 0, 'Inputs in every node must share the same size.'
    
    if rank == src:
        chunks = list(torch.chunk(x, world_size, dim=0))
    else:
        chunks = None
    x_chunk = torch.empty(
        len(x) // world_size, *x.shape[1:],
        dtype=x.dtype, device=x.device,
    )
    dist.scatter(x_chunk, chunks, src=src)
    return x_chunk


def gather(x: torch.Tensor) -> torch.Tensor:
    world_size = int(os.environ['WORLD_SIZE'])
    tensor_list = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(tensor_list, x)
    return torch.cat(tensor_list, dim=0)


def reduce(x: torch.Tensor, op: dist.ReduceOp=dist.ReduceOp.SUM) -> torch.Tensor:
    dist.all_reduce(x, op=op)
    return x
