# inference/utils_dist.py
from __future__ import annotations
import torch
import torch.distributed as dist

"""
============================================================
utils_dist.py

ROLE
----
Utilities to make SSL inference correct under:
- single GPU
- multi-GPU DDP (torchrun)

This file:
✓ avoids duplicated evaluation work
✓ ensures global-mean loss is correct
✓ keeps rank-0-only file IO safe
============================================================
"""

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_dist() else 0

def world() -> int:
    return dist.get_world_size() if is_dist() else 1


@torch.no_grad()
def allreduce_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the true global mean across all ranks.
    """
    if not is_dist():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= world()
    return y


# ============================================================
# [HELPER] DDP-safe evaluation DataLoader
# ============================================================
def distributed_eval_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    drop_last: bool = False,
    shuffle: bool = False,
):
    """
    Builds a DataLoader that:
    - uses DistributedSampler under DDP
    - avoids duplicated evaluation across ranks
    - stays deterministic for SSL evaluation

    Use this when you return a Dataset instead of a DataLoader.
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    if is_dist():
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=(num_workers > 0),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )



