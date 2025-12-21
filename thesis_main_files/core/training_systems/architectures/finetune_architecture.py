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
from core.NPVForensics.VACL_block.main.vacl_wrapper import VACLWrapper


@dataclass
class FinetuneArchitectureConfig:
    # ------------------------------------------------------------
    # [KEPT] Unifier dims
    # ------------------------------------------------------------
    vacl_s_out: int = 64
    vacl_d_v: int = 256
    vacl_d_a: int = 768

    # ------------------------------------------------------------
    # [KEPT] Forward flags
    # ------------------------------------------------------------
    compute_infonce: bool = True
    return_intermediates: bool = False

    # ------------------------------------------------------------
    # [ADDED] Future hook toggles
    # ------------------------------------------------------------
    enable_new_vacl_features: bool = False  # keep OFF until you paste the new wrapper feature API


class VACLFinetuneArchitecture(nn.Module):
    def __init__(
        self,
        *,
        video_backbone: nn.Module,
        audio_backbone: nn.Module,
        cfg: Optional[FinetuneArchitectureConfig] = None,
        c_v_in: int = 256,
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
        self.pre_vacl_unifier = PreVACLTokenUnifier(
            c_v_in=c_v_in,
            c_a_in=c_a_in,
            cfg=TokenUnifierForVACLConfig(
                s_out=self.cfg.vacl_s_out,
                d_v=self.cfg.vacl_d_v,
                d_a=self.cfg.vacl_d_a,
                n_heads=4,
                attn_dropout=0.0,
                proj_dropout=0.0,
                share_queries=False,
            ),
        )

        # ============================================================
        # [KEPT] VACL wrapper
        # IMPORTANT: return_intermediates is controlled via forward(...)
        # ============================================================
        self.vacl = VACLWrapper(return_intermediates=False)

        # ============================================================
        # [ADDED] Placeholder container for future feature adapter
        # (kept as Identity so it never breaks; you can replace later)
        # ============================================================
        self._new_feat_adapter = nn.Identity()

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
    def _extract_new_vacl_features_stub(self, vacl_out: Dict[str, Any]) -> Optional[Any]:
        if not bool(self.cfg.enable_new_vacl_features):
            return None

        # ------------------------------------------------------------
        # [STUB] Replace this with your real keys once you paste the new VACL wrapper
        #
        # Example patterns (DO NOT assume; placeholders only):
        #   feat = vacl_out["F_new"]                      # tensor
        #   feat = {"F": vacl_out["F"], "G": vacl_out["G"]}  # dict of tensors
        # ------------------------------------------------------------
        feat = None  # <-- keep None until the new wrapper keys are known

        # [STUB] Optional adaptation layer if you need shape/channel adjustment later
        if feat is not None:
            feat = self._new_feat_adapter(feat)

        return feat

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
            compute_infonce=compute_infonce,
            return_intermediates=return_intermediates,
        )

        # ============================================================
        # [ADDED][STUB-IN] new features from vacl_wrapper (non-breaking)
        # ============================================================
        new_vacl_features = self._extract_new_vacl_features_stub(vacl_out)

        # ============================================================
        # [PATCHED] Stage-2 standardized outputs
        # Must match system_fine.py expectations:
        #   "X_v_att", "X_a_att", "L_cor", "l_infonce"
        # ============================================================
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
        L_cor = vacl_out.get("L_cor", vacl_out.get("L_corr", vacl_out.get("loss_cor", None)))
        if L_cor is None:
            raise KeyError(
                "[VACLFinetuneArchitecture] VACL output missing correlation loss. "
                "Expected key: 'L_cor' (or compatible alias)."
            )

        # ------------------------------------------------------------
        # [REQUIRED] InfoNCE
        # ------------------------------------------------------------
        l_infonce = vacl_out.get("l_infonce", vacl_out.get("loss_infonce", vacl_out.get("loss_nce", None)))
        if l_infonce is None:
            # NOTE: your system_fine.py requires l_infonce, so we hard-fail here.
            raise KeyError(
                "[VACLFinetuneArchitecture] VACL output missing InfoNCE scalar. "
                "Expected key: 'l_infonce' (or compatible alias)."
            )

        out["X_v_att"] = X_v_att
        out["X_a_att"] = X_a_att
        out["L_cor"] = L_cor
        out["l_infonce"] = l_infonce

        # ------------------------------------------------------------
        # [ADDED] Optional: keep the new features around for later trainer/head usage
        # This does NOT affect current training because the system ignores this key.
        # ------------------------------------------------------------
        if new_vacl_features is not None:
            out["new_vacl_features"] = new_vacl_features

        return out
