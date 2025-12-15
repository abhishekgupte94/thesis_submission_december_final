# build_swin2d.py
# --------------------------------------------------------------------------------------
# Swin2D backbone builder for AUDIO (mel-spectrogram) that behaves like your Swin3D setup:
#
#  - No dataset / no trainer / no build_dataset.
#  - Build the official Microsoft SwinTransformer model via models.build.build_model().
#  - Return TOKENS only: (B, S, D)
#  - Ensure gradients flow end-to-end from external architecture losses.
#
# INPUT  : (B, 1, 96, 64)  (mel bins x time bins)
# OUTPUT : (B, S, D)       (token sequence, no pooling)
#
# --------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import sys
import torch
import torch.nn as nn

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required for build_swin2d.py because we load the Swin YAML config. "
        "Install with: pip install pyyaml"
    ) from e


# ======================================================================================
# 1) USER CONFIG: exactly matching what you confirmed
# ======================================================================================
@dataclass
class BuildSwin2DConfig:
    """
    Minimal config for Swin2D (official Microsoft Swin repo).

    IMPORTANT:
    - We will set IMG_SIZE=(96,64) and IN_CHANS=1 so PatchEmbed asserts pass.
    - We set NUM_CLASSES=0 so the head becomes Identity (but we don't call forward()).
    - We return tokens (B,S,D) directly after final norm, before avgpool/head.
    """
    # Repo import control
    # (We add external/Swin-Transformer onto sys.path at runtime, similar to your Swin3D fix.)
    use_syspath_inject: bool = True

    # Path to YAML config in your repo (you showed: external/Swin-Transformer/configs/swin/swin_base_patch4_window7_224.yaml)
    # You can pass this in when calling build_swin2d_backbone(...).
    yaml_path: Optional[str] = None

    # Audio input contract (you confirmed)
    img_size: Tuple[int, int] = (96, 64)  # (F, T) treated as (H, W)
    in_chans: int = 1

    # You confirmed embed_dim should be 128
    embed_dim: int = 128

    # Return tokens
    out: str = "tokens"  # only supported output for now: "tokens" => (B,S,D)

    # Optional: override depths/heads/window if you want to force them regardless of YAML
    # Leave as None to use YAML values (or defaults).
    depths: Optional[Sequence[int]] = None
    num_heads: Optional[Sequence[int]] = None
    window_size: Optional[int] = None

    # Optional: checkpointing (memory saving) — keep False for sanity on Mac
    use_checkpoint: bool = False


# ======================================================================================
# 2) PROJECT ROOT DISCOVERY (so this works from anywhere inside thesis_main_files)
# ======================================================================================
def _get_project_root(anchor: Optional[Path] = None) -> Path:
    """
    Walk upwards until we find a directory named 'thesis_main_files'.
    This matches your repo layout and avoids hardcoding absolute paths.
    """
    anchor = anchor or Path(__file__).resolve()
    for p in [anchor, *anchor.parents]:
        if p.name == "thesis_main_files":
            return p
    raise RuntimeError("Could not locate project root folder named 'thesis_main_files'.")


# ======================================================================================
# 3) IMPORT FIX: ensure official Swin repo is importable as a PACKAGE
# ======================================================================================
def _ensure_swin2d_repo_importable() -> None:
    """
    Ensures external/Swin-Transformer is importable so we can:
        from models.build import build_model
    without importing files by path (which breaks relative imports).

    NOTE:
    - This *does* inject into sys.path (same technique you ended up using for Swin3D).
    - It is the simplest robust approach if you are not doing `pip install -e external/Swin-Transformer`.
    """
    project_root = _get_project_root()
    swin_root = project_root / "external" / "Swin-Transformer"
    swin_root_str = str(swin_root)

    if swin_root_str not in sys.path:
        # Put it at front so the repo-local 'models', 'config', etc. resolve correctly.
        sys.path.insert(0, swin_root_str)


# ======================================================================================
# 4) YAML → "dot access" config object
#
# The official Swin build_model(config) expects an object with:
#   config.MODEL.TYPE
#   config.DATA.IMG_SIZE
#   config.MODEL.SWIN.PATCH_SIZE
#   config.MODEL.SWIN.IN_CHANS
#   config.MODEL.SWIN.EMBED_DIM
#   config.MODEL.SWIN.DEPTHS
#   config.MODEL.SWIN.NUM_HEADS
#   config.MODEL.SWIN.WINDOW_SIZE
#   config.MODEL.SWIN.MLP_RATIO
#   config.MODEL.SWIN.QKV_BIAS
#   config.MODEL.SWIN.QK_SCALE
#   config.MODEL.SWIN.APE
#   config.MODEL.SWIN.PATCH_NORM
#   config.MODEL.DROP_RATE
#   config.MODEL.DROP_PATH_RATE
#   config.MODEL.NUM_CLASSES
#   config.TRAIN.USE_CHECKPOINT
#   config.FUSED_LAYERNORM
#   config.FUSED_WINDOW_PROCESS
#
# Your pasted YAML snippet is partial (it doesn't include many of these).
# So: we load YAML (if provided) and fill missing fields with safe defaults.
# ======================================================================================
class _Attr:
    """
    Tiny recursive attribute dictionary so we can do config.MODEL.SWIN.EMBED_DIM, etc.
    """
    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            if isinstance(v, dict):
                v = _Attr(v)
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, _Attr) else v
        return out


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict, got {type(data)}")
    return data


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive dict merge: upd overrides base.
    """
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _make_swin2d_config(cfg: BuildSwin2DConfig) -> _Attr:
    """
    Build a minimal "official Swin repo config object" from YAML + our enforced overrides.
    """
    # --------------------------
    # Defaults that satisfy build_model() requirements
    # --------------------------
    defaults: Dict[str, Any] = {
        "DATA": {
            "IMG_SIZE": list(cfg.img_size),  # can be int or [H,W]; PatchEmbed uses to_2tuple
            "DATASET": "dummy",
        },
        "MODEL": {
            "TYPE": "swin",
            "NAME": "swin_audio_backbone",
            "NUM_CLASSES": 0,           # head becomes Identity (but we won't call forward anyway)
            "DROP_RATE": 0.0,
            "DROP_PATH_RATE": 0.2,
            "SWIN": {
                "PATCH_SIZE": 4,
                "IN_CHANS": cfg.in_chans,
                "EMBED_DIM": cfg.embed_dim,
                "DEPTHS": [2, 2, 18, 2],
                "NUM_HEADS": [4, 8, 16, 32],
                "WINDOW_SIZE": 7,
                "MLP_RATIO": 4.0,
                "QKV_BIAS": True,
                "QK_SCALE": None,
                "APE": False,
                "PATCH_NORM": True,
            },
        },
        "TRAIN": {
            "USE_CHECKPOINT": bool(cfg.use_checkpoint),
        },
        # build.py references these at top-level
        "FUSED_LAYERNORM": False,
        "FUSED_WINDOW_PROCESS": False,
    }

    # --------------------------
    # Load YAML (optional) and merge
    # --------------------------
    merged = defaults
    if cfg.yaml_path:
        y = _load_yaml(cfg.yaml_path)
        merged = _deep_update(merged, y)

    # --------------------------
    # Enforce your *hard requirements* regardless of YAML
    # --------------------------
    merged["DATA"]["IMG_SIZE"] = list(cfg.img_size)  # (96,64)
    merged["MODEL"]["TYPE"] = "swin"
    merged["MODEL"]["NUM_CLASSES"] = 0               # kill classifier head
    merged["MODEL"]["SWIN"]["IN_CHANS"] = cfg.in_chans
    merged["MODEL"]["SWIN"]["EMBED_DIM"] = cfg.embed_dim
    merged["TRAIN"]["USE_CHECKPOINT"] = bool(cfg.use_checkpoint)

    # Optional overrides if you want to force them from the builder call
    if cfg.depths is not None:
        merged["MODEL"]["SWIN"]["DEPTHS"] = list(cfg.depths)
    if cfg.num_heads is not None:
        merged["MODEL"]["SWIN"]["NUM_HEADS"] = list(cfg.num_heads)
    if cfg.window_size is not None:
        merged["MODEL"]["SWIN"]
