# wrapper_cpe.py
# ============================================================
# Common Projection Head Wrapper (CPE)
#
# FINAL VERSION:
#   - Original projection flow preserved
#   - InfoNCE computed internally (original semantics)
#   - Training-friendly loss-only default
#   - [ADDED] return_intermediates flag for debugging/analysis
# ============================================================

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# [FIXED] Correct import + naming
# ============================================================
# try:
from core.NPVForensics.common_projection.multimodal_projection_heads import MultiModalProjectionHeads



class FaceAudioCommonSpaceWrapper(nn.Module):
    """
    Face–Audio Common Projection Head Wrapper

    Default (training):
      - returns ONLY loss keys (DDP / Lightning friendly)

    Debug / analysis:
      - set return_intermediates=True to also return embeddings
    """

    def __init__(
        self,
        d_a: int,
        d_f: int,
        d_common: int,
        tau: float = 0.07,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.tau = float(tau)
        self.loss_weight = float(loss_weight)

        self.proj_heads = MultiModalProjectionHeads(
            d_a=d_a,
            d_f=d_f,
            d_common=d_common,
        )

    # ========================================================
    # [UNCHANGED] pooling logic
    # ========================================================
    def _ensure_pooled(
        self,
        X: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Ensures pooled (N, D) output.

        Supports:
          - (N, D)
          - (N, S, D) with optional masking
        """
        if X.dim() == 2:
            return X

        if X.dim() != 3:
            raise ValueError(f"Expected (N,D) or (N,S,D), got {tuple(X.shape)}")

        if lengths is None:
            return X.mean(dim=1)

        device = X.device
        N, S, _ = X.shape
        mask = (
            torch.arange(S, device=device)
            .unsqueeze(0)
            .expand(N, S)
            < lengths.unsqueeze(1)
        )
        mask = mask.unsqueeze(-1)
        return (X * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # ========================================================
    # [UNCHANGED] InfoNCE loss
    # ========================================================
    def _info_nce(self, Z_a: torch.Tensor, Z_f: torch.Tensor) -> torch.Tensor:
        """
        Symmetric face–audio InfoNCE loss.
        """
        Z_a = F.normalize(Z_a, dim=-1)
        Z_f = F.normalize(Z_f, dim=-1)

        logits = (Z_a @ Z_f.t()) / self.tau
        labels = torch.arange(Z_a.size(0), device=Z_a.device)

        loss_af = F.cross_entropy(logits, labels)
        loss_fa = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_af + loss_fa)

    # ========================================================
    # Forward
    # ========================================================
    def forward(
        self,
        audio_in: torch.Tensor,
        face_in: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        face_lengths: Optional[torch.Tensor] = None,
        compute_infonce: bool = True,
        # ----------------------------------------------------
        # [ADDED] intermediates control
        # ----------------------------------------------------
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          audio_in: (N,D) or (N,S,D)
          face_in:  (N,D) or (N,S,D)

        Returns:
          Always:
            - loss
            - loss_cpe
            - L_info

          If return_intermediates=True:
            - X_a, X_f
            - Z_a, Z_f
        """
        if not compute_infonce:
            raise ValueError(
                "compute_infonce=False is not supported in loss-only mode."
            )

        # ====================================================
        # Original projection flow
        # ====================================================
        X_a = self._ensure_pooled(audio_in, audio_lengths)
        X_f = self._ensure_pooled(face_in, face_lengths)

        proj_out = self.proj_heads(X_a=X_a, X_f=X_f)
        Z_a = proj_out["Z_a"]
        Z_f = proj_out["Z_f"]

        L_info = self._info_nce(Z_a, Z_f) * self.loss_weight

        # ====================================================
        # [TRAINING DEFAULT] loss-only
        # ====================================================
        out: Dict[str, torch.Tensor] = {
            # "L_info": L_info
            # "loss_cpe": L_info,
            "loss_cpe": L_info
        }

        # ====================================================
        # [ADDED] optional intermediates
        # ====================================================
        if return_intermediates:
            out.update(
                {
                    "X_a": X_a,
                    "X_f": X_f,
                    "Z_a": Z_a,
                    "Z_f": Z_f,
                }
            )

        return out
