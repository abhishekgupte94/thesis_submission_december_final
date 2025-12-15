

# pre_swin/video_pre_swin_encoder.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor


@dataclass
class VideoPreSwinConfig:
    image_size: int = 224
    max_frames: int = 32
    in_chans: int = 3
    embed_dim: int = 256
    patch_size: int = 4
    tubelet_size: int = 2
    pad_or_crop_time: bool = False


class LearnablePositionalEmbedding3D(nn.Module):
    """
    x:   (B, C, D, H_p, W_p)
    pos: (1, C, D_max, H_p_max, W_p_max)
    """
    def __init__(self, C: int, D_max: int, H_p_max: int, W_p_max: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, C, D_max, H_p_max, W_p_max) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"[PosEmb3D] expected (B,C,D,H,W), got {x.shape}")
        _, C, D, H, W = x.shape
        return x + self.pos[:, :C, :D, :H, :W]


class VideoPreSwinEncoder(nn.Module):
    """
    Pre-Swin video encoder (single stream).
    Output: (B, C, D, H_p, W_p)
    """
    def __init__(self, cfg: Optional[VideoPreSwinConfig] = None):
        super().__init__()
        self.cfg = cfg or VideoPreSwinConfig()

        assert self.cfg.image_size % self.cfg.patch_size == 0
        assert self.cfg.max_frames % self.cfg.tubelet_size == 0

        self.tubelet_embed = nn.Conv3d(
            in_channels=self.cfg.in_chans,
            out_channels=self.cfg.embed_dim,
            kernel_size=(self.cfg.tubelet_size, self.cfg.patch_size, self.cfg.patch_size),
            stride=(self.cfg.tubelet_size, self.cfg.patch_size, self.cfg.patch_size),
        )

        D_max = self.cfg.max_frames // self.cfg.tubelet_size
        H_p = self.cfg.image_size // self.cfg.patch_size
        W_p = self.cfg.image_size // self.cfg.patch_size
        self.pos3d = LearnablePositionalEmbedding3D(self.cfg.embed_dim, D_max, H_p, W_p)

    def _ensure_layout(self, clip: Tensor) -> Tensor:
        if clip.ndim == 4:  # (T,3,H,W)
            clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
        elif clip.ndim == 5 and clip.shape[2] == 3:  # (B,T,3,H,W)
            clip = clip.permute(0, 2, 1, 3, 4)
        elif clip.ndim != 5:
            raise ValueError(f"[VideoPreSwin] invalid shape {clip.shape}")
        return clip.contiguous()  # (B,3,T,H,W)

    def _ensure_time(self, clip: Tensor) -> Tensor:
        if not self.cfg.pad_or_crop_time:
            if clip.shape[2] != self.cfg.max_frames:
                raise ValueError(f"[VideoPreSwin] expected T={self.cfg.max_frames}, got {clip.shape[2]}")
            return clip

        B, C, T, H, W = clip.shape
        if T > self.cfg.max_frames:
            start = (T - self.cfg.max_frames) // 2
            return clip[:, :, start:start + self.cfg.max_frames]

        if T < self.cfg.max_frames:
            pad = self.cfg.max_frames - T
            last = clip[:, :, -1:, :, :].expand(B, C, pad, H, W)
            return torch.cat([clip, last], dim=2)

        return clip

    def forward(self, clip: Tensor) -> Tensor:
        clip = self._ensure_layout(clip)   # (B,3,T,H,W)
        clip = self._ensure_time(clip)     # (B,3,32,H,W)
        x = self.tubelet_embed(clip)       # (B,C,D,H_p,W_p)
        x = self.pos3d(x)
        return x
