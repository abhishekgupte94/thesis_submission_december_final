# backbones/video_backbone_swin3d.py

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# ============================================================
# Import your builder exports (do NOT edit build_swin3d.py)
# ============================================================
from build_swin3d import BuildSwin3DConfig, build_swin3d_backbone


class VideoBackboneSwin3D(nn.Module):
    """
    One wrapper that does EVERYTHING:
      - builds Swin3D backbone internally
      - accepts the video_batch directly
      - returns (B, C_v, T', H', W') from forward() and forward_features()

    Usage:
      video_backbone = VideoBackboneSwin3D(cfg)
      feat = video_backbone(video_batch)              # calls forward() -> forward_features()
      feat = video_backbone.forward_features(video_batch)
    """

    def __init__(self, cfg: BuildSwin3DConfig):
        super().__init__()
        self.cfg = cfg

        # [CRITICAL] downstream unifier expects 5D: (B,C,T',H',W')
        if getattr(self.cfg, "out", "5d") != "5d":
            raise ValueError(
                f"VideoBackboneSwin3D requires cfg.out='5d' but got out={getattr(self.cfg, 'out', None)!r}"
            )

        # ============================================================
        # [THE BUILDER CALL] happens HERE and only HERE
        # ============================================================
        self.backbone = build_swin3d_backbone(self.cfg)

    def _resolve_tensor(self, video_in: Any) -> torch.Tensor:
        """
        Supports:
          - Tensor input
          - Dict input containing the tensor under common keys
        """
        if torch.is_tensor(video_in):
            return video_in

        if isinstance(video_in, dict):
            for k in ("video", "pixel_values", "x", "frames"):
                if k in video_in:
                    t = video_in[k]
                    if not torch.is_tensor(t):
                        raise TypeError(f"video_in['{k}'] is not a Tensor (got {type(t)})")
                    return t
            raise KeyError(f"Video input dict has no known key. Keys={list(video_in.keys())}")

        raise TypeError(f"Expected Tensor or dict->Tensor, got {type(video_in)}")

    def forward_features(self, video_in: Any) -> torch.Tensor:
        x = self._resolve_tensor(video_in)
        return self.backbone.forward_features(x)  # (B, C_v, T', H', W')

    def forward(self, video_in: Any) -> torch.Tensor:
        # Make the module callable like a normal backbone
        return self.forward_features(video_in)
