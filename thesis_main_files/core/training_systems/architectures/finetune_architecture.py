# core/training_systems/architectures/vacl_finetune_architecture.py
# ============================================================
# [FINAL PATCHED] VACLFinetuneArchitecture
#
# Purpose:
#   Stage-2 finetune architecture for AVPretrainSystem (LightningModule).
#
# Contract (REQUIRED by system_fine.py):
#   forward(...) returns dict with:
#       "X_v_att":   (B, k, S)
#       "X_a_att":   (B, k, S)
#       "L_cor":     scalar
#       "l_infonce": scalar  (REQUIRED)
#
# Notes:
#   - Pure nn.Module (no Lightning)
#   - No logging, no device placement
#   - DDP-safe, bf16-mixed safe
#   - Swin backbones are frozen by AVPretrainSystem (not here)
#
# IMPORTANT CHANGE REQUESTED:
#   - Model will take "new features" from vacl_wrapper.
#   - We add a stub-in hook to extract/transform those features later.
#   - The stub is NON-BREAKING and does not change the required outputs.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ============================================================
# [KEPT] Token unifier (same as pretrain)
# ============================================================
from scripts.feature_extraction.token_unifier_post_swin.token_unifier_post_swin import (
    PreVACLTokenUnifier,
    TokenUnifierForVACLConfig,
)

# ============================================================
# [KEPT] VACL wrapper import
# ============================================================
from core.NPVForensics.VACL_block.main.vacl_wrapper_fine_tune import VACLWrapper
from core.NPVForensics.common_projection.main.common_projection_head_module_wrapper import (
    FaceAudioCommonSpaceWrapper
    # FaceAudioCommonProjectionConfig,
)

@dataclass
class FinetuneArchitectureConfig:
    # ------------------------------------------------------------
    # [KEPT] Unifier dims
    # ------------------------------------------------------------
    vacl_s_out: int = 64
    vacl_d_v: int = 768
    vacl_d_a: int = 768

    # ------------------------------------------------------------
    # [KEPT] Forward flags
    # ------------------------------------------------------------
    compute_infonce: bool = True
    return_intermediates: bool = False
    cpe_d_common: int = 512

    # ------------------------------------------------------------
    # [ADDED] Future hook toggles
    # ------------------------------------------------------------


class FinetuneArchitecture(nn.Module):
    def __init__(
        self,
        *,
        video_backbone: nn.Module,
        audio_backbone: nn.Module,
        cfg: Optional[FinetuneArchitectureConfig] = None,
        c_v_in: int = 768,
        c_a_in: int = 768,
    ) -> None:
        super().__init__()
        self.cfg = FinetuneArchitectureConfig() if cfg is None else cfg

        # ============================================================
        # [KEPT] Injected backbones (frozen by system_fine.py)
        # ============================================================
        self.video_backbone = video_backbone
        self.audio_backbone = audio_backbone

        # ============================================================
        # [KEPT] Pre-VACL token unifier
        # ============================================================

        ## TOKENIZER NEEDS TO BE FROZEN IN THE FINE TUNE ARCHITECTURE
        self.pre_vacl_unifier = PreVACLTokenUnifier(
            c_v_in=c_v_in,  # e.g., 256
            c_a_in=c_a_in,  # e.g., 768
            cfg=TokenUnifierForVACLConfig(
                interp_mode="linear",
                align_corners=False,
                grid_hw=(7, 7),
                audio_tokens=49,
            ),
        )

        # ============================================================
        # [KEPT] VACL wrapper
        # IMPORTANT: return_intermediates is controlled via forward(...)
        # ============================================================
        ### CONFIG NEEDS TO BE ADDED LIKE pretrain_architecture.py

        self.vacl = VACLWrapper(
            vacl_kwargs=dict(
                d_v=cfg.vacl_d_v,
                d_a=cfg.vacl_d_a,
                seq_len=cfg.vacl_s_out,
                k=64, ),
            return_intermediates=False
        )

        self.common_proj = FaceAudioCommonSpaceWrapper(
            d_a=cfg.vacl_d_a,  # audio feature dimension
            d_f=cfg.vacl_d_a,  # face / video feature dimension
            d_common=cfg.cpe_d_common,  # shared embedding dimension
            tau=0.07,  # temperature for InfoNCE
            loss_weight=1.0
        )

        # self.prb_model = PRBModel(...)


    # ============================================================
    # [ADDED][STUB-IN] New features hook from VACL wrapper
    #
    # Requirement from you:
    #   "the model will take new features from the vacl_wrapper"
    #
    # Implementation policy:
    #   - MUST be non-breaking today (Stage-2 system expects only X_v_att, X_a_att, L_cor, l_infonce)
    #   - Provide a clean insertion point for later when you paste the new wrapper feature API
    #
    # How to use later:
    #   - Change this function to read the correct keys from vacl_out
    #   - Return a tensor or dict of tensors, then consume them downstream
    # ============================================================

    def forward(
        self,
        *,
        video_in: torch.Tensor,
        audio_in: torch.Tensor,
        compute_infonce: Optional[bool] = None,
        return_intermediates: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # ------------------------------------------------------------
        # [KEPT] Resolve flags
        # ------------------------------------------------------------
        if compute_infonce is None:
            compute_infonce = self.cfg.compute_infonce
        if return_intermediates is None:
            return_intermediates = self.cfg.return_intermediates
        # ============================================================
        # [ADDED][PATCH] Video dtype fix for bf16-mixed:
        #   dataloader returns uint8 frames (0..255) for mp4 decode.
        #   Swin3D expects floating input. Cast + normalize here.
        #   Keep device placement to Lightning (no .to(device) calls).
        # ============================================================
        if video_in.dtype == torch.uint8:
            video_in = video_in.float().div_(255.0)
        elif not torch.is_floating_point(video_in):
            video_in = video_in.float()

        # Optional (only if your Swin3D was trained with ImageNet stats):
        # mean/std for RGB (channels-first)
        # mean = torch.tensor([0.485, 0.456, 0.406], device=video_in.device).view(1, 3, 1, 1, 1)
        # std  = torch.tensor([0.229, 0.224, 0.225], device=video_in.device).view(1, 3, 1, 1, 1)
        # video_in = (video_in - mean) / std

        # ============================================================
        # [KEPT] Backbone forward
        # ============================================================
        # video_backbone: expected to expose forward_features(video_in)
        # audio_backbone: expected to expose forward_features(audio_in)
        video_feat_3d = self.video_backbone.forward_features(video_in)
        audio_tokens = self.audio_backbone.forward_features(audio_in)

        # ============================================================
        # [KEPT] Token unification (produces X_v, X_a expected by VACL)
        # ============================================================
        X_v, X_a = self.pre_vacl_unifier(
            video_feat_3d=video_feat_3d,
            audio_tokens=audio_tokens,
        )

        # ============================================================
        # [KEPT] VACL forward
        # ============================================================
        vacl_out = self.vacl(
            X_v=X_v,
            X_a=X_a,
            # compute_infonce=compute_infonce,
            return_intermediates=return_intermediates,
        )
        cpe_out = self.common_proj(
            X_v=X_v,
            X_a=X_a,
            # compute_infonce=compute_infonce,
            return_intermediates=return_intermediates,
        )


        out: Dict[str, Any] = {}


        # ------------------------------------------------------------
        # [REQUIRED] Attention maps
        # ------------------------------------------------------------
        # Prefer explicit keys if your wrapper uses them; fall back safely.
        X_v_att = vacl_out.get("X_v_att", vacl_out.get("x_v_att", None))
        X_a_att = vacl_out.get("X_a_att", vacl_out.get("x_a_att", None))
        if X_v_att is None or X_a_att is None:
            raise KeyError(
                "[VACLFinetuneArchitecture] VACL output missing attention maps. "
                "Expected keys: 'X_v_att' and 'X_a_att' (case-sensitive)."
            )

        # ------------------------------------------------------------
        # [REQUIRED] Correlation loss
        # ------------------------------------------------------------
        L_cor = vacl_out.get("loss_vacl")
        if L_cor is None:
            raise KeyError(
                "[VACLFinetuneArchitecture] VACL output missing correlation loss. "
                "Expected key: 'L_cor' (or compatible alias)."
            )

        loss_cpe = cpe_out.get("loss_cpe", cpe_out.get("loss", None))

        out["X_v_att"] = X_v_att
        out["X_a_att"] = X_a_att
        out["L_cor"] = L_cor
        out["l_infonce"] = loss_cpe

        # ------------------------------------------------------------
        # [ADDED] Optional: keep the new features around for later trainer/head usage
        # This does NOT affect current training because the system ignores this key.
        # ------------------------------------------------------------
        # if new_vacl_features is not None:
        #     out["new_vacl_features"] = new_vacl_features

        return out
