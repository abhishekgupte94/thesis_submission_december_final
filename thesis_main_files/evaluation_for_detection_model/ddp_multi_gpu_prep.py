# evaluation_for_detection_model/utils_dist.py
from __future__ import annotations
import torch
import torch.distributed as dist

"""
============================================================
utils_dist.py

ROLE
----
Minimal utilities to keep evaluation_for_detection_model:
✓ single-GPU safe
✓ multi-GPU DDP safe

Used ONLY for:
- rank detection
- world size
- mean aggregation
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
    Computes mean across all ranks.
    Required for SSL loss reporting.
    """
    if not is_dist():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= world()
    return y
