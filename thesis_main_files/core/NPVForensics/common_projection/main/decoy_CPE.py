
# common_projection_head_module_wrapper.py
# ============================================================
# [DROP-IN] CPE (Common Projection / EC Mining) Wrapper
#
# Purpose:
#   - Wraps the Common Space / Projection Head stack
#   - Makes outputs consistent with Lightning / DDP
#   - Ensures:
#       • no logits returned unless explicitly requested
#       • memory-heavy intermediates are stripped by default
#
# Expected inputs:
#   X_v : Tensor (B, D_v, S) or (B, S, D_v)
#   X_a : Tensor (B, D_a, S) or (B, S, D_a)
#
# Outputs (dict):
#   - loss_cpe : scalar Tensor
#
# Notes:
#   - No device placement here
#   - No logging here
#   - Safe for bf16 / DDP
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ============================================================
# [EXISTING] Import your actual projection head modules
# ============================================================
from common_space_projector import CommonSpaceProjector
from multimodal_projection_heads import MultimodalProjectionHeads


# ============================================================
# [ADDED] Configuration container
# ============================================================
@dataclass
class FaceAudioCommonProjectionConfig:
    # behaviour flags
    return_intermediates: bool = False
    strip_intermediates: bool = True
    expect_bds: bool = True        # expect (B,D,S)

    # loss weighting
    loss_weight: float = 1.0

    # projector / head configs (keep your real fields here)
    # e.g.:
    # d_v: int = 256
    # d_a: int = 768
    # d_common: int = 256
    # temperature: float = 0.07
    # ...


class FaceAudioCommonSpaceWrapper(nn.Module):
    def __init__(
        self,
        cfg: FaceAudioCommonProjectionConfig,
        *,
        projector: Optional[nn.Module] = None,
        heads: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # ============================================================
        # [MODIFIED] Allow injection or internal construction
        # ============================================================
        self.projector = projector or CommonSpaceProjector(
            # >>> KEEP YOUR ORIGINAL ARGS HERE <<<
        )

        self.heads = heads or MultimodalProjectionHeads(
            # >>> KEEP YOUR ORIGINAL ARGS HERE <<<
        )

    # ============================================================
    # [ADDED] Layout normalizer
    # ============================================================
    @staticmethod
    def _to_expected_layout(x: torch.Tensor, expect_bds: bool) -> torch.Tensor:
        """
        Normalizes tensor layout:
          expect_bds=True  -> (B,D,S)
          expect_bds=False -> (B,S,D)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}")

        B, A, C = x.shape

        if expect_bds:
            # want (B,D,S)
            return x.transpose(1, 2).contiguous() if A < C else x
        else:
            # want (B,S,D)
            return x.transpose(1, 2).contiguous() if A > C else x

    # ============================================================
    # [FORWARD]
    # ============================================================
    def forward(
        self,
        *,
        X_v: torch.Tensor,
        X_a: torch.Tensor,
        compute_infonce: bool = True,
        return_intermediates: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "loss_cpe": Tensor scalar
          }
        """
        if return_intermediates is None:
            return_intermediates = bool(self.cfg.return_intermediates)

        # ============================================================
        # [ADDED] Normalize layout
        # ============================================================
        X_v = self._to_expected_layout(X_v, self.cfg.expect_bds)
        X_a = self._to_expected_layout(X_a, self.cfg.expect_bds)

        # ============================================================
        # [ADDED] Enforce same S
        # ============================================================
        if self.cfg.expect_bds:
            if X_v.shape[2] != X_a.shape[2]:
                raise ValueError(f"CPE expects same S. Got Sv={X_v.shape[2]} Sa={X_a.shape[2]}")
        else:
            if X_v.shape[1] != X_a.shape[1]:
                raise ValueError(f"CPE expects same S. Got Sv={X_v.shape[1]} Sa={X_a.shape[1]}")

        # ============================================================
        # [KEPT] Project to common space
        # ============================================================
        Z_v, Z_a = self.projector(X_v=X_v, X_a=X_a)

        # ============================================================
        # [KEPT] Compute contrastive / EC loss
        # ============================================================
        heads_out = self.heads(
            Z_v=Z_v,
            Z_a=Z_a,
            compute_infonce=bool(compute_infonce),
            return_intermediates=bool(return_intermediates),
            **kwargs,
        )

        if not isinstance(heads_out, dict):
            if isinstance(heads_out, torch.Tensor):
                return {"loss_cpe": self.cfg.loss_weight * heads_out}
            raise TypeError(f"Projection heads returned unexpected type: {type(heads_out)}")

        loss = heads_out.get("loss_cpe", heads_out.get("loss", None))
        if loss is None:
            raise KeyError(
                "Projection heads output missing loss key "
                "['loss_cpe','loss']"
            )

        out: Dict[str, Any] = {
            "loss_cpe": self.cfg.loss_weight * loss
        }

        # ============================================================
        # [ADDED] Strip heavy tensors unless requested
        # ============================================================
        if not return_intermediates and self.cfg.strip_intermediates:
            for k in [
                "logits", "logits_va", "logits_av",
                "similarity", "sim_matrix",
                "intermediates", "attn",
            ]:
                heads_out.pop(k, None)

        if return_intermediates:
            out["intermediates"] = heads_out

        return out