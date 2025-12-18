from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any
from pathlib import Path
import sys

import torch
import torch.nn as nn


# ======================================================================================
# Config
# ======================================================================================
@dataclass
class Swin2DAudioWrapperConfig:
    # Swin repo YAML (optional)
    yaml_path: Optional[str] = None

    # Checkpoint (optional)
    ckpt_path: Optional[str] = None
    ckpt_prefix_strip: Optional[str] = None  # e.g. "module.", "backbone."

    # Input mel expected from your AudioPreprocessorNPV
    mel_hw: Tuple[int, int] = (64, 96)  # (H,W)

    # Swin-friendly target
    out_hw: Tuple[int, int] = (224, 224)
    to_3ch: bool = True
    normalize: str = "per_sample"   # "none" | "per_sample"
    pad_value: str = "min"          # "min" | float
    time_align: str = "left"        # "left" | "center"

    # Force Swin to be built in vanilla mode
    swin_img_size: Tuple[int, int] = (224, 224)
    swin_in_chans: int = 3


# ======================================================================================
# Swin import wiring (same as you had)
# ======================================================================================
def _get_project_root(anchor: Optional[Path] = None) -> Path:
    anchor = anchor or Path(__file__).resolve()
    for p in [anchor, *anchor.parents]:
        if p.name == "thesis_main_files":
            return p
    raise RuntimeError("Could not locate project root folder named 'thesis_main_files'.")


def _ensure_swin2d_on_syspath() -> None:
    project_root = _get_project_root()
    swin_root = project_root / "external" / "Swin-Transformer"
    s = str(swin_root)
    if s not in sys.path:
        sys.path.insert(0, s)


# ======================================================================================
# (Steps 1–3) Adapt mel to Swin-friendly format
# ======================================================================================
def mel_b1_64x96_to_swin_input(
    mel: torch.Tensor,
    *,
    out_hw=(224, 224),
    to_3ch=True,
    normalize="per_sample",
    eps=1e-6,
    pad_value="min",
    time_align="left",
) -> torch.Tensor:
    # shape -> (B,1,H,W)
    if mel.ndim == 2:
        mel = mel.unsqueeze(0).unsqueeze(0)
    elif mel.ndim == 3:
        if mel.shape[0] == 1:
            mel = mel.unsqueeze(0)
        else:
            mel = mel.unsqueeze(1)
    elif mel.ndim != 4:
        raise ValueError(f"Unexpected mel shape: {tuple(mel.shape)}")

    B, C, H, W = mel.shape
    if C != 1:
        raise ValueError(f"Expected C=1 mel, got {C} for shape {tuple(mel.shape)}")

    # fix common swap
    if (H, W) == (96, 64):
        mel = mel.transpose(2, 3).contiguous()

    # normalize
    if normalize == "per_sample":
        mean = mel.mean(dim=(2, 3), keepdim=True)
        std = mel.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        mel = (mel - mean) / std
    elif normalize != "none":
        raise ValueError("normalize must be 'none' or 'per_sample'")

    # pad/crop into (OH,OW)
    OH, OW = out_hw
    mel = mel[:, :, :min(mel.shape[2], OH), :min(mel.shape[3], OW)]
    _, _, Hc, Wc = mel.shape

    if pad_value == "min":
        fill = mel.amin(dim=(2, 3), keepdim=True)
    elif isinstance(pad_value, (int, float)):
        fill = torch.tensor(float(pad_value), device=mel.device, dtype=mel.dtype).view(1, 1, 1, 1)
    else:
        raise ValueError("pad_value must be 'min' or float")

    canvas = fill.expand(B, 1, OH, OW).clone()

    if time_align == "left":
        top = (OH - Hc) // 2
        left = 0
    elif time_align == "center":
        top = (OH - Hc) // 2
        left = (OW - Wc) // 2
    else:
        raise ValueError("time_align must be 'left' or 'center'")

    canvas[:, :, top:top + Hc, left:left + Wc] = mel
    mel = canvas

    if to_3ch:
        mel = mel.repeat(1, 3, 1, 1)

    return mel


# ======================================================================================
# Checkpoint helpers
# ======================================================================================
def _load_ckpt_state_dict(ckpt_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")


def _strip_prefix(state: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items() }


# ======================================================================================
# The wrapper
# ======================================================================================
class Swin2DAudioBackboneWrapper(nn.Module):
    """
    Trainer-facing wrapper.

    Input:  mel (B,1,64,96) from AudioPreprocessorNPV
    Output: tokens (B,L,C) from Swin2D.forward_features()

    It builds the Swin2D model internally and performs the Swin-friendly conversion.
    """
    def __init__(self, cfg: Swin2DAudioWrapperConfig):
        super().__init__()
        self.cfg = cfg

        _ensure_swin2d_on_syspath()
        from models.build import build_model  # external/Swin-Transformer

        # Build a config object for build_model. Reuse your previous _Attr approach:
        # We'll create only what build_model needs.
        class _Attr:
            def __init__(self, d: dict):
                for k, v in d.items():
                    setattr(self, k, _Attr(v) if isinstance(v, dict) else v)

        base = {
            "DATA": {"IMG_SIZE": list(cfg.swin_img_size), "DATASET": "dummy"},
            "MODEL": {
                "TYPE": "swin",
                "NAME": "swin_tiny_patch4_window7_224",
                "NUM_CLASSES": 0,
                "DROP_RATE": 0.0,
                "DROP_PATH_RATE": 0.2,
                "SWIN": {
                    "PATCH_SIZE": 4,
                    "IN_CHANS": int(cfg.swin_in_chans),   # 3
                    "EMBED_DIM": 96,
                    "DEPTHS": [2, 2, 6, 2],
                    "NUM_HEADS": [3, 6, 12, 24],
                    "WINDOW_SIZE": 7,
                    "MLP_RATIO": 4.0,
                    "QKV_BIAS": True,
                    "QK_SCALE": None,
                    "APE": False,
                    "PATCH_NORM": True,
                },
            },
            "TRAIN": {"USE_CHECKPOINT": False},
            "FUSED_LAYERNORM": False,
            "FUSED_WINDOW_PROCESS": False,
        }

        # Optional YAML overlay (if you want)
        if cfg.yaml_path:
            import yaml
            with open(cfg.yaml_path, "r") as f:
                y = yaml.safe_load(f) or {}

            def deep_update(a, b):
                out = dict(a)
                for k, v in b.items():
                    if isinstance(v, dict) and isinstance(out.get(k), dict):
                        out[k] = deep_update(out[k], v)
                    else:
                        out[k] = v
                return out

            base = deep_update(base, y)

        config_obj = _Attr(base)

        self.backbone = build_model(config_obj, is_pretrain=False)

        # Optional ckpt
        if cfg.ckpt_path:
            state = _load_ckpt_state_dict(cfg.ckpt_path, map_location="cpu")
            if cfg.ckpt_prefix_strip:
                state = _strip_prefix(state, cfg.ckpt_prefix_strip)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[Swin2DAudioBackboneWrapper] Loaded {cfg.ckpt_path} strict=False")
                if missing:
                    print(f"  Missing keys (sample): {missing[:20]}")
                if unexpected:
                    print(f"  Unexpected keys (sample): {unexpected[:20]}")

    def forward_features(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel_b1_64x96_to_swin_input(
            mel,
            out_hw=self.cfg.out_hw,
            to_3ch=self.cfg.to_3ch,
            normalize=self.cfg.normalize,
            pad_value=self.cfg.pad_value,
            time_align=self.cfg.time_align,
        )
        return self.backbone.forward_features(x)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.forward_features(mel)
