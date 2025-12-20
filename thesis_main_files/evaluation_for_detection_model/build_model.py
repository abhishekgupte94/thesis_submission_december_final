
from __future__ import annotations

"""
============================================================
build_model.py

ROLE
----
Rebuilds the *exact SSL model topology* used during training,
without assuming any external pretrained checkpoints.

Key facts (your setup):
-----------------------
- Swin2D and Swin3D are trained ON THE FLY
- best_weights.pt contains ALL learned weights
- No classification heads exist
- Forward returns ONLY SSL losses

This file is used by:
- min_eval_ssl_losses.py
- any future probing / export scripts
============================================================
"""

from dataclasses import dataclass
from typing import Tuple
import torch

# === Core SSL architecture ===
from core.training_systems.architectures.pretrain_architecture import AVPretrainArchitecture, ArchitectureConfig

# === Swin wrappers (must match training) ===
from scripts.feature_extraction.SWIN.main.MAIN_swin2d_wrapper import (
    Swin2DAudioBackboneWrapper,
    Swin2DAudioWrapperConfig,
)
from scripts.feature_extraction.SWIN.main.MAIN_swin_3d_wrapper import VideoBackboneSwin3D
from scripts.feature_extraction.SWIN.main.build_swin3d import BuildSwin3DConfig


# ============================================================
# [CONFIG] Minimal build args for inference
# ============================================================
@dataclass
class BuildModelArgs:
    device: str = "cuda"
    freeze_backbones: bool = True


# ============================================================
# [HELPER] Freeze modules for inference safety
# ============================================================
def _freeze_module(m: torch.nn.Module) -> None:
    """
    Disables gradients and switches to eval mode.

    Prevents:
    - accidental finetuning
    - batchnorm / dropout drift
    """
    for p in m.parameters():
        p.requires_grad = False
    m.eval()


# ============================================================
# [CORE] Build SSL model topology (no weights loaded here)
# ============================================================
def build_model(args: BuildModelArgs) -> AVPretrainArchitecture:
    """
    Constructs the SSL architecture exactly as in training.

    Returns:
        AVPretrainArchitecture
        (forward() → dict of SSL losses only)
    """

    # -------- Audio backbone (Swin2D) --------
    audio_backbone = Swin2DAudioBackboneWrapper(
        Swin2DAudioWrapperConfig(
            # Defaults must match training
        )
    )

    # -------- Video backbone (Swin3D) --------
    swin3d_cfg = BuildSwin3DConfig(
        out="5d",
        use_checkpoint=True,   # same activation checkpointing as training
    )
    video_backbone = VideoBackboneSwin3D(swin3d_cfg)

    if args.freeze_backbones:
        _freeze_module(audio_backbone)
        _freeze_module(video_backbone)

    # -------- SSL architecture config --------
    arch_cfg = ArchitectureConfig(
        vacl_s_out=64,
        vacl_d_v=256,
        vacl_d_a=768,
        compute_infonce=True,
        return_intermediates=False,
        lambda_vacl=1.0,
        lambda_cpe=1.0,
    )

    model = AVPretrainArchitecture(
        cfg=arch_cfg,
        video_backbone=video_backbone,
        audio_backbone=audio_backbone,
        c_v_in=256,
        c_a_in=768,
    )

    model.to(args.device)
    model.eval()
    return model


# ============================================================
# [LOAD] Load trained weights (includes Swins + SSL heads)
# ============================================================
def load_weights_into_model(
    model: torch.nn.Module,
    weights_pt: str,
    strict: bool = True,
) -> Tuple[int, int]:
    """
    Loads weights saved by AVPretrainSystem:

        payload["state_dict"] = model.state_dict()

    This includes:
    - Swin2D weights
    - Swin3D weights
    - VACL parameters
    - InfoNCE / CPE parameters
    """
    payload = torch.load(weights_pt, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=strict)

    return int(payload.get("epoch", -1)), int(payload.get("global_step", -1))




