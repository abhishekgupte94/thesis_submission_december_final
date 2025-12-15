# common_space_projector.py

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CommonSpaceProjector(nn.Module):
    """
    Projection head mapping (N,D) -> (N,D_common).

    We keep the same inductive bias used in the paper's projection heads:
      - num_layers=1: Linear -> BN
      - num_layers=2: Linear -> BN -> ReLU -> Linear -> BN

    NOTE:
      - This module expects pooled inputs (N,D).
      - Pooling from (N,S,D) is handled in the wrapper.
    """

    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 1):
        super().__init__()
        if num_layers not in (1, 2):
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)

        if self.num_layers == 1:
            self.fc = nn.Linear(self.in_dim, self.out_dim, bias=True)
            self.bn = nn.BatchNorm1d(self.out_dim)
        else:
            # two-layer + ReLU with BN after each linear
            self.fc1 = nn.Linear(self.in_dim, self.out_dim, bias=True)
            self.bn1 = nn.BatchNorm1d(self.out_dim)
            self.fc2 = nn.Linear(self.out_dim, self.out_dim, bias=True)
            self.bn2 = nn.BatchNorm1d(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, D_in)
        returns: (N, D_out)
        """
        if x.dim() != 2:
            raise ValueError(f"CommonSpaceProjector expects (N,D). Got {tuple(x.shape)}")

        if self.num_layers == 1:
            return self.bn(self.fc(x))

        h = self.bn1(self.fc1(x))
        h = F.relu(h, inplace=False)
        y = self.bn2(self.fc2(h))
        return y
