from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple
import sys

import torch
import torch.nn as nn

try:
    import yaml
except ImportError as e:
    raise ImportError("Please install PyYAML: pip install pyyaml") from e


# --------------------------------------------------------------------------------------
# Config for Swin2D audio backbone
# --------------------------------------------------------------------------------------
@dataclass
class BuildSwin2DConfig:
    # Optional YAML (your Swin repo config). If provided, we load & overlay defaults.
    yaml_path: Optional[str] = None

    # ----------------------------------------------------------------------------------
    # [MODIFIED] Default to **Swin-Tiny** geometry.
    # Reason: you asked to patch the builder config to the canonical tiny settings:
    #   NAME: swin_tiny_patch4_window7_224
    #   EMBED_DIM: 96
    #   DEPTHS: [2, 2, 6, 2]
    #   NUM_HEADS: [3, 6, 12, 24]
    #   WINDOW_SIZE: 7
    #
    # Note: we also default IMG_SIZE to 224x224 to match the tiny config name.
    # For audio, the sanity script will resize a log-mel image to (224,224).
    # ----------------------------------------------------------------------------------
    img_size: Tuple[int, int] = (224, 224)
    in_chans: int = 1
    embed_dim: int = 96

    # Optional overrides (if you don’t want to rely on YAML for these)
    # [MODIFIED] Defaults set to tiny; still overridable if you pass explicit args.
    depths: Optional[Sequence[int]] = (2, 2, 6, 2)
    num_heads: Optional[Sequence[int]] = (3, 6, 12, 24)
    window_size: Optional[int] = 7

    use_checkpoint: bool = False


# --------------------------------------------------------------------------------------
# Project root + sys.path wiring (same idea as your Swin3D builder)
# --------------------------------------------------------------------------------------
def _get_project_root(anchor: Optional[Path] = None) -> Path:
    anchor = anchor or Path(__file__).resolve()
    for p in [anchor, *anchor.parents]:
        if p.name == "thesis_main_files":
            return p
    raise RuntimeError("Could not locate project root folder named 'thesis_main_files'.")


def _ensure_swin2d_on_syspath() -> None:
    """
    Make external/Swin-Transformer importable so we can do:
        from models.build import build_model
    """
    project_root = _get_project_root()
    swin_root = project_root / "external" / "Swin-Transformer"
    swin_root_str = str(swin_root)
    if swin_root_str not in sys.path:
        sys.path.insert(0, swin_root_str)


# --------------------------------------------------------------------------------------
# Adapter: call ONLY forward_features() and return BxLxC tokens
# --------------------------------------------------------------------------------------
class Swin2DTokenAdapter(nn.Module):
    """
    Adapter for Swin2D used as a token backbone.

    Assumption:
      - You patched SwinTransformer.forward_features() to return tokens (B, L, C)
        after self.norm, and NOT perform avgpool/head.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Calls official backbone path; no re-implementation here.
        tokens = self.backbone.forward_features(x)  # (B, L, C)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Safe to use in your pipeline (your external arch calls adapter(x))
        return self.forward_features(x)


# --------------------------------------------------------------------------------------
# YAML helpers
# --------------------------------------------------------------------------------------
def _load_yaml_dict(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict, got {type(data)}")
    return data


def _deep_update(base: dict, upd: dict) -> dict:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


class _Attr:
    """Recursive dot-access wrapper (so build_model can use config.MODEL.SWIN...)."""
    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                v = _Attr(v)
            setattr(self, k, v)


def _make_config_obj(cfg: BuildSwin2DConfig) -> _Attr:
    """
    build_model(config) expects a structured config with many keys.
    We provide safe defaults, then overlay YAML (if any), then enforce your constraints.
    """
    defaults = {
        "DATA": {"IMG_SIZE": list(cfg.img_size), "DATASET": "dummy"},
        "MODEL": {
            "TYPE": "swin",
            # ----------------------------------------------------------------------
            # [MODIFIED] Match canonical Swin-Tiny naming.
            # Reason: your builder can be swapped with pretrained checkpoints/configs
            # that key off this name.
            # ----------------------------------------------------------------------
            "NAME": "swin_tiny_patch4_window7_224",
            "NUM_CLASSES": 0,          # irrelevant because we won't call forward()/head
            "DROP_RATE": 0.0,
            "DROP_PATH_RATE": 0.2,
            "SWIN": {
                # ------------------------------------------------------------------
                # [MODIFIED] Tiny uses patch4.
                # Reason: your requested config is swin_tiny_patch4_window7_224.
                # ------------------------------------------------------------------
                "PATCH_SIZE": 4,
                "IN_CHANS": cfg.in_chans,
                "EMBED_DIM": cfg.embed_dim,
                # ------------------------------------------------------------------
                # [MODIFIED] Tiny stage depths & heads.
                # Reason: exactly the values you provided.
                # ------------------------------------------------------------------
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
        "TRAIN": {"USE_CHECKPOINT": bool(cfg.use_checkpoint)},
        "FUSED_LAYERNORM": False,
        "FUSED_WINDOW_PROCESS": False,
    }

    merged = defaults
    if cfg.yaml_path:
        merged = _deep_update(merged, _load_yaml_dict(cfg.yaml_path))

    # Enforce your confirmed constraints (do not trust YAML to match audio)
    merged["DATA"]["IMG_SIZE"] = list(cfg.img_size)
    merged["MODEL"]["SWIN"]["IN_CHANS"] = cfg.in_chans
    merged["MODEL"]["SWIN"]["EMBED_DIM"] = cfg.embed_dim
    merged["TRAIN"]["USE_CHECKPOINT"] = bool(cfg.use_checkpoint)

    if cfg.depths is not None:
        merged["MODEL"]["SWIN"]["DEPTHS"] = list(cfg.depths)
    if cfg.num_heads is not None:
        merged["MODEL"]["SWIN"]["NUM_HEADS"] = list(cfg.num_heads)
    if cfg.window_size is not None:
        merged["MODEL"]["SWIN"]["WINDOW_SIZE"] = int(cfg.window_size)

    return _Attr(merged)


# --------------------------------------------------------------------------------------
# Public builder
# --------------------------------------------------------------------------------------
def build_swin2d_backbone(cfg: Optional[BuildSwin2DConfig] = None) -> nn.Module:
    cfg = cfg or BuildSwin2DConfig()

    # Ensure the official Swin repo is importable
    _ensure_swin2d_on_syspath()

    # Import official builder from the Swin repo
    from models.build import build_model  # from external/Swin-Transformer/models/build.py

    # Build config object matching builder expectations
    config_obj = _make_config_obj(cfg)

    # Construct official SwinTransformer model
    backbone = build_model(config_obj, is_pretrain=False)

    # Wrap in token adapter (calls forward_features only)
    return Swin2DTokenAdapter(backbone)
