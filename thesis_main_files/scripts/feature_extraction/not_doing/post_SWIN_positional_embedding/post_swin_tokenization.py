from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# ============================================================
# Output container (exactly what VACL expects)
# ============================================================
@dataclass
class VACLAlignedTokens:
    X_a: Tensor   # (B, S, d_a)
    X_v: Tensor   # (B, S, d_v)
    X_f: Tensor   # (B, S, d_f)
    S: int


# ============================================================
# Post-Swin temporal alignment module
# ============================================================
class PostSwinTemporalEncoder(nn.Module):
    """
    AFTER Swin encoders, BEFORE VACL.

    Audio (Swin-2D):
        (B, C_a, H_a, W_a)
          -> mean over H_a
          -> (B, W_a, C_a)
          -> resample to S

    Video (Swin-3D):
        (B, C_v, D, H, W)
          -> mean over (H,W)
          -> (B, D, C_v)

    S := D (video temporal depth)
    """

    def __init__(self, resample_mode: str = "linear"):
        super().__init__()
        self.resample_mode = resample_mode

    # -------------------------
    # Video → temporal tokens
    # -------------------------
    @staticmethod
    def _video_to_temporal_tokens(grid_bcdhw: Tensor) -> Tensor:
        """
        (B, C, D, H, W) → (B, D, C)
        """
        if grid_bcdhw.ndim != 5:
            raise ValueError(f"Expected video grid (B,C,D,H,W), got {grid_bcdhw.shape}")

        return (
            grid_bcdhw
            .mean(dim=(3, 4))          # (B, C, D)
            .permute(0, 2, 1)          # (B, D, C)
            .contiguous()
        )

    # -------------------------
    # Audio → time tokens
    # -------------------------
    @staticmethod
    def _audio_to_time_tokens(grid_bchw: Tensor) -> Tensor:
        """
        (B, C, H_a, W_a) → (B, W_a, C)
        """
        if grid_bchw.ndim != 4:
            raise ValueError(f"Expected audio grid (B,C,H,W), got {grid_bchw.shape}")

        return (
            grid_bchw
            .mean(dim=2)               # pool frequency H_a → (B, C, W_a)
            .permute(0, 2, 1)          # (B, W_a, C)
            .contiguous()
        )

    # -------------------------
    # Time resampling
    # -------------------------
    def _resample_time(self, x_btc: Tensor, target_S: int) -> Tensor:
        """
        (B, T0, C) → (B, target_S, C)
        """
        B, T0, C = x_btc.shape
        if T0 == target_S:
            return x_btc

        x = x_btc.permute(0, 2, 1)  # (B, C, T0)
        x = F.interpolate(
            x,
            size=target_S,
            mode=self.resample_mode,
            align_corners=False if self.resample_mode == "linear" else None,
        )
        return x.permute(0, 2, 1).contiguous()

    # -------------------------
    # Forward
    # -------------------------
    def forward(
        self,
        audio_grid_bchw: Tensor,
        viseme_grid_bcdhw: Tensor,
        face_grid_bcdhw: Tensor,
    ) -> VACLAlignedTokens:

        # Video → temporal tokens
        X_v = self._video_to_temporal_tokens(viseme_grid_bcdhw)  # (B, D, d_v)
        X_f = self._video_to_temporal_tokens(face_grid_bcdhw)    # (B, D, d_f)

        S = X_v.shape[1]  # shared sequence length

        # Audio → time tokens → resample to S
        X_a0 = self._audio_to_time_tokens(audio_grid_bchw)       # (B, W_a, d_a)
        X_a = self._resample_time(X_a0, target_S=S)              # (B, S, d_a)

        return VACLAlignedTokens(X_a=X_a, X_v=X_v, X_f=X_f, S=S)
