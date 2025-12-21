from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import sys
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Config (matches VST SwinTransformer3D signature)
# --------------------------------------------------------------------------------------
@dataclass
class BuildSwin3DConfig:
    pretrained: Optional[str] = None
    pretrained2d: bool = True

    patch_size: Tuple[int, int, int] = (4, 4, 4)
    in_chans: int = 3
    embed_dim: int = 96

    depths: Sequence[int] = (2, 2, 6, 2)
    num_heads: Sequence[int] = (3, 6, 12, 24)
    window_size: Tuple[int, int, int] = (2, 7, 7)

    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None

    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.2

    patch_norm: bool = False
    frozen_stages: int = -1
    use_checkpoint: bool = False

    out: str = "5d"


# --------------------------------------------------------------------------------------
# Project root helper
# --------------------------------------------------------------------------------------
def _get_project_root(anchor: Optional[Path] = None) -> Path:
    anchor = anchor or Path(__file__).resolve()
    for p in [anchor, *anchor.parents]:
        if p.name == "thesis_main_files":
            return p
    raise RuntimeError("Could not locate project root folder named 'thesis_main_files'.")


# --------------------------------------------------------------------------------------
# Ensure VST repo is importable as a PACKAGE (critical fix)
# --------------------------------------------------------------------------------------
def _ensure_vst_on_syspath() -> None:
    """
    Robust: find a folder under the current working directory that contains mmaction/
    and add its parent to sys.path.
    """
    cwd = Path.cwd().resolve()

    # Look for mmaction directory anywhere under cwd
    for mmaction_dir in cwd.rglob("mmaction"):
        if mmaction_dir.is_dir() and (mmaction_dir / "models").exists():
            vst_root = mmaction_dir.parent
            vst_root_str = str(vst_root)
            if vst_root_str not in sys.path:
                sys.path.insert(0, vst_root_str)
            return

    raise RuntimeError(f"Could not find a repo containing mmaction/ under: {cwd}")



class Swin3DFeatureAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, out: str = "5d"):
        super().__init__()
        self.backbone = backbone
        self.out = out

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        if self.out == "5d":
            return y
        if self.out == "bd":
            return y.mean(dim=(2, 3, 4))
        raise ValueError(f"Unknown out='{self.out}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def build_swin3d_backbone(
    cfg: Optional[BuildSwin3DConfig] = None,
) -> nn.Module:
    cfg = cfg or BuildSwin3DConfig()

    # 🔑 Critical: ensure correct import context
    _ensure_vst_on_syspath()

    # Import SWIN as a PACKAGE module (relative imports now work)
    from mmaction.models.backbones.swin_transformer import SwinTransformer3D

    backbone = SwinTransformer3D(
        pretrained=cfg.pretrained,
        pretrained2d=cfg.pretrained2d,
        patch_size=tuple(cfg.patch_size),
        in_chans=cfg.in_chans,
        embed_dim=cfg.embed_dim,
        depths=list(cfg.depths),
        num_heads=list(cfg.num_heads),
        window_size=tuple(cfg.window_size),
        mlp_ratio=cfg.mlp_ratio,
        qkv_bias=cfg.qkv_bias,
        qk_scale=cfg.qk_scale,
        drop_rate=cfg.drop_rate,
        attn_drop_rate=cfg.attn_drop_rate,
        drop_path_rate=cfg.drop_path_rate,
        patch_norm=cfg.patch_norm,
        frozen_stages=cfg.frozen_stages,
        use_checkpoint=cfg.use_checkpoint,
    )

    if hasattr(backbone, "init_weights"):
        backbone.init_weights(pretrained=cfg.pretrained)

    return Swin3DFeatureAdapter(backbone, out=cfg.out)
