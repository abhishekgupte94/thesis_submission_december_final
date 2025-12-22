# multimodal_projection_heads.py

from __future__ import annotations

import torch
from torch import nn

from core.NPVForensics.common_projection.common_space_projector import CommonSpaceProjector


class MultiModalProjectionHeads(nn.Module):
    """
    Face + Audio projection heads into a shared common space (S_fa).

    You said you're using ONLY face-audio InfoNCE.
    So we define:
      - g_a_to_fa : audio -> common space  (Linear+BN)
      - g_f_to_fa : face  -> common space  (2-layer MLP + ReLU + BN)

    Inputs must be pooled to (N,D) before calling these heads.
    """

    def __init__(self, d_a: int, d_f: int, d_common: int):
        super().__init__()
        self.d_a = int(d_a)
        self.d_f = int(d_f)
        self.d_common = int(d_common)

        # Audio projector (simpler head)
        self.g_a_to_fa = CommonSpaceProjector(in_dim=self.d_a, out_dim=self.d_common, num_layers=1)

        # Face projector (deeper head)
        self.g_f_to_fa = CommonSpaceProjector(in_dim=self.d_f, out_dim=self.d_common, num_layers=2)

    @staticmethod
    def _ensure_bd(x: torch.Tensor, *, name: str) -> torch.Tensor:
        """
        Accept:
          - (B, D)    -> unchanged
          - (B, S, D) -> mean over S -> (B, D)
        """
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            # IMPORTANT: pool over sequence dim, not embedding dim
            return x.mean(dim=1)
        raise ValueError(f"[CPE] {name} expected (B,D) or (B,S,D), got {tuple(x.shape)}")
    def forward(self, X_a: torch.Tensor, X_f: torch.Tensor) -> dict:
        """
        X_a: (N, d_a)
        X_f: (N, d_f)

        Returns:
          Z_a: (N, d_common)
          Z_f: (N, d_common)
        """
        X_a = self._ensure_bd(X_a, name="X_a")  # (B, Da)
        X_f = self._ensure_bd(X_f, name="X_f")  # (B, Df)

        if X_a.dim() != 2 or X_f.dim() != 2:
            raise ValueError(f"ProjectionHeads expect pooled (N,D). Got {X_a.shape}, {X_f.shape}")

        Z_a = self.g_a_to_fa(X_a)
        Z_f = self.g_f_to_fa(X_f)
        return {"Z_a": Z_a, "Z_f": Z_f}
