# pretrain_architecture.py
# ============================================================
# [FINAL PATCHED] AVPretrainArchitecture
#
# Pure nn.Module:
#   - No logging
#   - No device placement
#   - No Lightning dependencies
#
# Outputs:
#   - loss_total
#   - loss_vacl
#   - loss_cpe
#
# Safe for:
#   - DDP
#   - bf16-mixed
#   - SSL training
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

# ============================================================
# [KEPT] Token unifier
# ============================================================
from scripts.feature_extraction.token_unifier_post_swin.token_unifier_post_swin import (
    PreVACLTokenUnifier,
    TokenUnifierForVACLConfig,
)

# ============================================================
# [MODIFIED] Use VACLWrapper instead of raw block
# ============================================================
from core.NPVForensics.VACL_block.main.vacl_wrapper import VACLWrapper

# ============================================================
# [KEPT] Common projection / CPE wrapper
# ============================================================
from core.NPVForensics.common_projection.main.common_projection_head_module_wrapper import (
    FaceAudioCommonSpaceWrapper
    # FaceAudioCommonProjectionConfig,
)


@dataclass
class ArchitectureConfig:
    vacl_s_out: int = 64
    vacl_d_v: int = 768
    vacl_d_a: int = 768

    cpe_d_common: int = 512
    compute_infonce: bool = True
    return_intermediates: bool = False

    lambda_vacl: float = 1.0
    lambda_cpe: float = 0.1


class AVPretrainArchitecture(nn.Module):
    def __init__(
        self,
        cfg: ArchitectureConfig,
        video_backbone: nn.Module,
        audio_backbone: nn.Module,
        c_v_in: int = 256,
        c_a_in: int = 768,
    ):
        super().__init__()
        self.cfg = cfg

        # ============================================================
        # [KEPT] Injected backbones (trainable)
        # ============================================================
        self.video_backbone = video_backbone
        self.audio_backbone = audio_backbone

        # ============================================================
        # [KEPT] Pre-VACL token unifier
        # ============================================================
        self.pre_vacl_unifier = PreVACLTokenUnifier(
            c_v_in=c_v_in,
            c_a_in=c_a_in,
            cfg=TokenUnifierForVACLConfig(
                s_out=cfg.vacl_s_out,
                d_v=cfg.vacl_d_v,
                d_a=cfg.vacl_d_a,
                n_heads=4,
                attn_dropout=0.2,
                proj_dropout=0.1,
                share_queries=False,
            ),
        )

        # ============================================================
        # [MODIFIED] VACL wrapper
        # ============================================================
        self.vacl = VACLWrapper(
            vacl_kwargs=dict(
                d_v=cfg.vacl_d_v,
                d_a=cfg.vacl_d_a,
                seq_len=cfg.vacl_s_out,
                k=64,),
            return_intermediates=False
        )

        # ============================================================
        # [KEPT] Common projection (CPE)
        # ============================================================
        self.common_proj = FaceAudioCommonSpaceWrapper(
            d_a=cfg.vacl_d_a,  # audio feature dimension
            d_f=cfg.vacl_d_a,  # face / video feature dimension
            d_common=cfg.cpe_d_common,  # shared embedding dimension
            tau=0.07,  # temperature for InfoNCE
            loss_weight=1.0
        )

    def forward(
        self,
        *,
        video_in: torch.Tensor,
        audio_in: torch.Tensor,
        compute_infonce: Optional[bool] = None,
        return_intermediates: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        if compute_infonce is None:
            compute_infonce = self.cfg.compute_infonce
        if return_intermediates is None:
            return_intermediates = self.cfg.return_intermediates

        # ============================================================
        # [KEPT] Backbone forward
        # ============================================================
        video_feat_3d = self.video_backbone.forward_features(video_in)
        audio_tokens = self.audio_backbone.forward_features(audio_in)

        # ============================================================
        # [KEPT] Token unification
        # ============================================================
        X_v, X_a = self.pre_vacl_unifier(
            video_feat_3d=video_feat_3d,
            audio_tokens=audio_tokens,
        )

        # ============================================================
        # [KEPT] VACL
        # ============================================================
        vacl_out = self.vacl(
            X_v=X_v,
            X_a=X_a,
            # compute_infonce=compute_infonce,
            return_intermediates=return_intermediates,
        )

        # ============================================================
        # [KEPT] CPE
        # ============================================================
        cpe_out = self.common_proj(
            X_v=X_v,
            X_a=X_a,
            # compute_infonce=compute_infonce,
            return_intermediates=return_intermediates,
        )

        # ============================================================
        # [PATCHED] Standardize outputs
        # ============================================================
        out: Dict[str, Any] = {}

        loss_vacl = vacl_out.get("loss_vacl")
        loss_cpe = cpe_out.get("loss_cpe")

        out["loss_vacl"] = loss_vacl
        out["loss_cpe"] = loss_cpe

        total = 0.0
        if loss_vacl is not None:
            total = total + self.cfg.lambda_vacl * loss_vacl
        if loss_cpe is not None:
            total = total + self.cfg.lambda_cpe * loss_cpe

        out["loss_total"] = total

        return out