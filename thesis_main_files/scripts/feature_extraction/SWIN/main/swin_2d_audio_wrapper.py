from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _ensure_audio_4d(x: torch.Tensor, layout: str) -> torch.Tensor:
    """
    Normalise audio tensor to (B, C, F, T).

    Supported layouts:
      - "BFT"  : (B, F, T)       -> (B, 1, F, T)
      - "BTF"  : (B, T, F)       -> (B, 1, F, T)
      - "BCFT" : (B, C, F, T)    -> unchanged
      - "B1FT" : (B, 1, F, T)    -> unchanged
    """
    if layout == "BFT":
        if x.ndim != 3:
            raise ValueError(f"BFT expects (B,F,T); got {tuple(x.shape)}")
        return x.unsqueeze(1)

    if layout == "BTF":
        if x.ndim != 3:
            raise ValueError(f"BTF expects (B,T,F); got {tuple(x.shape)}")
        return x.transpose(1, 2).unsqueeze(1)

    if layout in ("BCFT", "B1FT"):
        if x.ndim != 4:
            raise ValueError(f"{layout} expects (B,C,F,T); got {tuple(x.shape)}")
        return x

    raise ValueError(f"Unknown audio layout={layout!r} (use BFT, BTF, BCFT, B1FT).")


def _as_feature_tensor(out: Any) -> torch.Tensor:
    """
    Swin repos sometimes return Tensor, tuple(Tensor, ...), or dict.
    We normalise to a single Tensor.
    """
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
class Swin2DAudioWrapperConfig:
    input_layout: str = "BFT"          # your raw: (B, n_mel, T)
    feature_call: str = "auto"         # "auto" | "forward_features" | "forward"


class Swin2DAudioFeatureWrapper(nn.Module):
    """
    Wraps a Swin2D backbone for AUDIO features.

    Input:
      - (B, n_mel, T) if input_layout="BFT"
      - or (B, 1, n_mel, T) if input_layout="B1FT"/"BCFT"

    Output dict:
      {
        "feat": Tensor,
        "meta": {...}
      }
    """
    def __init__(self, backbone: nn.Module, cfg: Optional[Swin2DAudioWrapperConfig] = None):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg or Swin2DAudioWrapperConfig()

    def forward(self, audio: torch.Tensor) -> Dict[str, Any]:
        x = _ensure_audio_4d(audio, self.cfg.input_layout)

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
                "wrapper": "Swin2DAudioFeatureWrapper",
                "input_layout": self.cfg.input_layout,
                "feature_call": self.cfg.feature_call,
                "backbone": self.backbone.__class__.__name__,
                "input_shape": tuple(audio.shape),
                "norm_shape": tuple(x.shape),
                "feat_shape": tuple(feat.shape),
            },
        }
