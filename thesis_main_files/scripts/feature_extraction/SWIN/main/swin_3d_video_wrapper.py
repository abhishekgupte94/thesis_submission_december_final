from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _ensure_video_5d(x: torch.Tensor, layout: str) -> torch.Tensor:
    """
    Normalise video tensor to (B, C, T, H, W).

    Supported layouts:
      - "BCTHW": (B, C, T, H, W) -> unchanged
      - "BTCHW": (B, T, C, H, W) -> permuted -> (B, C, T, H, W)
    """
    if x.ndim != 5:
        raise ValueError(f"Video expects 5D, got {tuple(x.shape)}")

    if layout == "BCTHW":
        return x

    if layout == "BTCHW":
        return x.permute(0, 2, 1, 3, 4).contiguous()

    raise ValueError(f"Unknown video layout={layout!r} (use BCTHW or BTCHW).")


def _as_feature_tensor(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out

    if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
        return out[0]

    if isinstance(out, dict):
        for k in ("feat", "features", "x", "out", "last_hidden_state"):
            v = out.get(k, None)
            if torch.is_tensor(v):
                return v
        for v in out.values():
            if torch.is_tensor(v):
                return v

    raise TypeError(f"Unsupported backbone output type: {type(out)}")


@dataclass
class Swin3DVideoWrapperConfig:
    input_layout: str = "BCTHW"        # your raw: (B, 3, T, H, W)
    feature_call: str = "auto"         # "auto" | "forward_features" | "forward"


class Swin3DVideoFeatureWrapper(nn.Module):
    """
    Wraps a Video Swin (Swin3D) backbone for VIDEO features.

    Input:
      - (B, 3, T, H, W) if input_layout="BCTHW"
      - or (B, T, 3, H, W) if input_layout="BTCHW"

    Output dict:
      {
        "feat": Tensor,
        "meta": {...}
      }
    """
    def __init__(self, backbone: nn.Module, cfg: Optional[Swin3DVideoWrapperConfig] = None):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg or Swin3DVideoWrapperConfig()

    def forward(self, video: torch.Tensor) -> Dict[str, Any]:
        x = _ensure_video_5d(video, self.cfg.input_layout)

        if self.cfg.feature_call == "forward_features":
            if not hasattr(self.backbone, "forward_features"):
                raise AttributeError("Backbone has no forward_features()")
            out = self.backbone.forward_features(x)  # type: ignore[attr-defined]
        elif self.cfg.feature_call == "forward":
            out = self.backbone(x)
        elif self.cfg.feature_call == "auto":
            ff = getattr(self.backbone, "forward_features", None)
            out = ff(x) if callable(ff) else self.backbone(x)
        else:
            raise ValueError(f"Unknown feature_call={self.cfg.feature_call!r}")

        feat = _as_feature_tensor(out)

        return {
            "feat": feat,
            "meta": {
                "wrapper": "Swin3DVideoFeatureWrapper",
                "input_layout": self.cfg.input_layout,
                "feature_call": self.cfg.feature_call,
                "backbone": self.backbone.__class__.__name__,
                "input_shape": tuple(video.shape),
                "norm_shape": tuple(x.shape),
                "feat_shape": tuple(feat.shape),
            },
        }
