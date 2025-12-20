# evaluation_for_detection_model/build_model.py
from __future__ import annotations

"""
============================================================
build_model.py  (NO external Swin checkpoints)

ROLE (SSL CONTEXT)
-----------------
Builds the *same topology* as training:
- Swin2D wrapper (audio) with default config
- Swin3D wrapper (video) with BuildSwin3DConfig defaults used in training
- AVPretrainArchitecture

Then (elsewhere) you load best_weights.pt which contains ALL weights:
✓ Swin2D
✓ Swin3D
✓ VACL/CPE stacks
============================================================
"""

from dataclasses import dataclass
from typing import Tuple
import torch

from pretrain_architecture import AVPretrainArchitecture, ArchitectureConfig

from scripts.feature_extraction.SWIN.main.MAIN_swin2d_wrapper import (
    Swin2DAudioBackboneWrapper,
    Swin2DAudioWrapperConfig,
)
from scripts.feature_extraction.SWIN.main.MAIN_swin_3d_wrapper import VideoBackboneSwin3D
from scripts.feature_extraction.SWIN.main.build_swin3d import BuildSwin3DConfig


@dataclass
class BuildModelArgs:
    device: str = "cuda"
    freeze_backbones: bool = True   # keep True for evaluation_for_detection_model to avoid accidental grads


def _freeze_module(m: torch.nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False
    m.eval()


def build_model(args: BuildModelArgs) -> AVPretrainArchitecture:
    # ------------------------------------------------------------
    # [MATCH TRAINING] Build Swin2D audio backbone (default config)
    # ------------------------------------------------------------
    audio_backbone = Swin2DAudioBackboneWrapper(
        Swin2DAudioWrapperConfig(
            # Keep defaults (as in your main_trainer_pretrain.py)
        )
    )

    # ------------------------------------------------------------
    # [MATCH TRAINING] Build Swin3D video backbone
    # NOTE: fields must match your BuildSwin3DConfig dataclass
    # ------------------------------------------------------------
    swin3d_cfg = BuildSwin3DConfig(
        out="5d",
        use_checkpoint=True,  # activation memory saver (same as training)
        # add other required fields only if your builder requires them
    )
    video_backbone = VideoBackboneSwin3D(swin3d_cfg)

    if args.freeze_backbones:
        _freeze_module(audio_backbone)
        _freeze_module(video_backbone)

    # ------------------------------------------------------------
    # [ARCH CFG] SSL-only objectives
    # ------------------------------------------------------------
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


def load_weights_into_model(
    model: torch.nn.Module,
    weights_pt: str,
    strict: bool = True,
) -> Tuple[int, int]:
    """
    Loads checkpoints/best_weights.pt saved by your AVPretrainSystem:
        payload["state_dict"] = self.model.state_dict()

    This includes Swin2D/Swin3D weights (trained on the fly).
    """
    payload = torch.load(weights_pt, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=strict)
    return int(payload.get("epoch", -1)), int(payload.get("global_step", -1))
