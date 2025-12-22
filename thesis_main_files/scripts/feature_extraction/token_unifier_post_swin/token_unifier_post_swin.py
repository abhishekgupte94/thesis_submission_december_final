from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TokenUnifierForVACLConfig:
    # Keep this so your init code doesn't change much
    # Only the fields we actually need for the simple tokenizer:
    interp_mode: str = "linear"         # "linear" or "nearest"
    align_corners: bool = False
    grid_hw: Tuple[int, int] = (7, 7)   # 49 tokens -> 7x7
    audio_tokens: int = 49              # token count


class PreVACLTokenUnifier(nn.Module):
    """
    [SIMPLE DROP-IN] PreVACLTokenUnifier

    Inputs:
      - video_feat_3d: (B, Dv, T', 7, 7)
      - audio_tokens : (B, 49, Da)

    Outputs:
      - X_v: (B, T', Dv)   (spatial mean pool)
      - X_a: (B, T', Da)   (49 -> 7x7 -> pool over "freq" -> 7 bins -> interpolate to T')
    """

    def __init__(self, *, c_v_in: int, c_a_in: int, cfg: TokenUnifierForVACLConfig):
        super().__init__()
        self.c_v_in = c_v_in
        self.c_a_in = c_a_in
        self.cfg = cfg

    def forward(self, *, video_feat_3d: torch.Tensor, audio_tokens: torch.Tensor):
        # -------------------------
        # Video: (B,Dv,T,7,7) -> (B,T,Dv)
        # -------------------------
        if video_feat_3d.ndim != 5:
            raise ValueError(f"Expected video_feat_3d (B,Dv,T,H,W), got {tuple(video_feat_3d.shape)}")
        Bv, Dv, T, H, W = video_feat_3d.shape
        if Dv != self.c_v_in:
            # not fatal, but helps catch wiring mistakes
            raise ValueError(f"video_feat_3d Dv={Dv} != c_v_in={self.c_v_in}")

        X_v = video_feat_3d.mean(dim=(3, 4)).transpose(1, 2).contiguous()  # (B,T,Dv)

        # -------------------------
        # Audio: (B,49,Da) -> (B,T,Da)
        # -------------------------
        if audio_tokens.ndim != 3:
            raise ValueError(f"Expected audio_tokens (B,49,Da), got {tuple(audio_tokens.shape)}")
        Ba, N, Da = audio_tokens.shape
        if Ba != Bv:
            raise ValueError(f"Batch mismatch: audio B={Ba} vs video B={Bv}")
        if Da != self.c_a_in:
            raise ValueError(f"audio_tokens Da={Da} != c_a_in={self.c_a_in}")

        Hh, Ww = self.cfg.grid_hw
        if N != self.cfg.audio_tokens or N != Hh * Ww:
            raise ValueError(f"Expected audio token count {Hh*Ww}, got N={N}")

        # (B,49,Da) -> (B,Da,7,7)
        a_map = audio_tokens.transpose(1, 2).reshape(Ba, Da, Hh, Ww).contiguous()

        # pool "freq" axis -> (B,Da,7)  (treat W as time bins)
        a_time7 = a_map.mean(dim=2)  # (B,Da,Ww=7)

        # interpolate 7 -> T
        a_timeT = F.interpolate(
            a_time7,
            size=T,
            mode=self.cfg.interp_mode,
            align_corners=self.cfg.align_corners if self.cfg.interp_mode in ("linear", "bilinear", "bicubic", "trilinear") else None,
        )  # (B,Da,T)

        X_a = a_timeT.transpose(1, 2).contiguous()  # (B,T,Da)

        return X_v, X_a
