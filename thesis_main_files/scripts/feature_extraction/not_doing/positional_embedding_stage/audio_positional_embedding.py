# pre_swin/audio_pre_swin_encoder.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclass
class AudioPreSwinConfig:
    n_mels: int = 96
    time_bins: int = 64
    in_chans: int = 1
    embed_dim: int = 256
    patch_size: int = 4
    pad_or_crop_time: bool = False


class LearnablePositionalEmbedding2D(nn.Module):
    """
    x:   (B, C, H_p, W_p)
    pos: (1, C, H_p_max, W_p_max)
    """
    def __init__(self, C: int, H_p_max: int, W_p_max: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, C, H_p_max, W_p_max) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"[PosEmb2D] expected (B,C,H,W), got {x.shape}")
        _, C, H, W = x.shape
        return x + self.pos[:, :C, :H, :W]


class AudioPreSwinEncoder(nn.Module):
    """
    Pre-Swin audio encoder.
    Output: (B, C, H_p, W_p)
    """
    def __init__(self, cfg: Optional[AudioPreSwinConfig] = None):
        super().__init__()
        self.cfg = cfg or AudioPreSwinConfig()

        assert self.cfg.n_mels % self.cfg.patch_size == 0
        assert self.cfg.time_bins % self.cfg.patch_size == 0

        self.patch_embed = nn.Conv2d(
            in_channels=self.cfg.in_chans,
            out_channels=self.cfg.embed_dim,
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        )

        H_p = self.cfg.n_mels // self.cfg.patch_size
        W_p = self.cfg.time_bins // self.cfg.patch_size
        self.pos2d = LearnablePositionalEmbedding2D(self.cfg.embed_dim, H_p, W_p)

    def _ensure_shape(self, mel: Tensor) -> Tensor:
        if mel.ndim == 2:
            mel = mel.unsqueeze(0).unsqueeze(0)
        elif mel.ndim == 3:
            mel = mel.unsqueeze(1)
        elif mel.ndim != 4:
            raise ValueError(f"[AudioPreSwin] invalid shape {mel.shape}")

        if not self.cfg.pad_or_crop_time and mel.shape[3] != self.cfg.time_bins:
            raise ValueError(f"[AudioPreSwin] expected T={self.cfg.time_bins}, got {mel.shape[3]}")

        if self.cfg.pad_or_crop_time and mel.shape[3] != self.cfg.time_bins:
            T = mel.shape[3]
            if T > self.cfg.time_bins:
                start = (T - self.cfg.time_bins) // 2
                mel = mel[:, :, :, start:start + self.cfg.time_bins]
            else:
                mel = F.pad(mel, (0, self.cfg.time_bins - T))

        return mel

    def forward(self, mel: Tensor) -> Tensor:
        mel = self._ensure_shape(mel)
        x = self.patch_embed(mel)   # (B,C,H_p,W_p)
        x = self.pos2d(x)
        return x