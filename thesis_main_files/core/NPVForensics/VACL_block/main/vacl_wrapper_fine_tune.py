# vacl_wrapper_fine_tune.py
# ============================================================
# [DROP-IN][PATCHED] VACLWrapper (Stage-2 compatible)
#
# Stage-2 REQUIRED output keys ensured:
#   - "X_v_att": (B,k,S)
#   - "X_a_att": (B,k,S)
#   - "L_cor":   scalar tensor
#
# Notes:
#   - No device moves
#   - No logging
#   - DDP/Lightning-friendly
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from core.NPVForensics.VACL_block.vacl_block import VACLVA


@dataclass
class VACLWrapperConfig:
    return_intermediates: bool = False
    strip_intermediates: bool = True
    expect_bds: bool = True  # True expects (B,D,S); False expects (B,S,D)


class VACLWrapper(nn.Module):
    def __init__(
        self,
        vacl: Optional[VACLVA] = None,
        vacl_kwargs: Optional[Dict[str, Any]] = None,
        cfg: Optional[VACLWrapperConfig] = None,
        return_intermediates: Optional[bool] = None,
        strip_intermediates: Optional[bool] = None,
        expect_bds: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.cfg = cfg or VACLWrapperConfig()

        # [ADDED] overrides
        if return_intermediates is not None:
            self.cfg.return_intermediates = bool(return_intermediates)
        if strip_intermediates is not None:
            self.cfg.strip_intermediates = bool(strip_intermediates)
        if expect_bds is not None:
            self.cfg.expect_bds = bool(expect_bds)

        # [KEPT] build underlying
        if vacl is not None:
            self.vacl = vacl
        else:
            vacl_kwargs = vacl_kwargs or {}
            self.vacl = VACLVA(**vacl_kwargs)

    @staticmethod
    def _to_expected_layout(x: torch.Tensor, expect_bds: bool) -> torch.Tensor:
        """
        If expect_bds=True, ensures shape (B,D,S).
        If expect_bds=False, ensures shape (B,S,D).
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B,*,*), got shape {tuple(x.shape)}")

        B, A, C = x.shape

        if expect_bds:
            # want (B,D,S); if (B,S,D) -> transpose
            return x.transpose(1, 2).contiguous() if A < C else x
        else:
            # want (B,S,D); if (B,D,S) -> transpose
            return x.transpose(1, 2).contiguous() if A > C else x

    def forward(
        self,
        *,
        X_v: torch.Tensor,
        X_a: torch.Tensor,
        return_intermediates: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        if return_intermediates is None:
            return_intermediates = bool(self.cfg.return_intermediates)

        # [KEPT] Normalize layout
        X_v = self._to_expected_layout(X_v, expect_bds=self.cfg.expect_bds)
        X_a = self._to_expected_layout(X_a, expect_bds=self.cfg.expect_bds)

        # [KEPT] Check same S
        if self.cfg.expect_bds:
            if X_v.shape[2] != X_a.shape[2]:
                raise ValueError(f"VACL expects same S. Got Sv={X_v.shape[2]} Sa={X_a.shape[2]}")
        else:
            if X_v.shape[1] != X_a.shape[1]:
                raise ValueError(f"VACL expects same S. Got Sv={X_v.shape[1]} Sa={X_a.shape[1]}")

        # ============================================================
        # [PATCHED] Call underlying VACLVA
        # - Some implementations accept compute_infonce, some don't.
        # - We try the richer signature first, then fallback safely.
        # ============================================================
        vacl_out = self.vacl(
            X_v=X_v,
            X_a=X_a,
            # compute_infonce=bool(compute_infonce),
            return_intermediates=False,
            **kwargs,
        )


        # ============================================================
        # [KEPT] Normalize output container
        # ============================================================
        if not isinstance(vacl_out, dict):
            if isinstance(vacl_out, torch.Tensor):
                # [PATCHED] Stage-2 keys: L_cor + l_infonce ensured
                z = vacl_out.new_zeros(())
                return {
                    "loss_vacl": vacl_out,
                    # "loss": vacl_out,
                    # "L_cor": vacl_out
                    # "l_infonce": z,
                }
            raise TypeError(f"VACLVA returned unexpected type: {type(vacl_out)}")

        out: Dict[str, Any] = dict(vacl_out)

        # ============================================================
        # [PATCHED] Standardize required Stage-2 keys
        # ============================================================
        # ---- loss / L_cor
        L_cor = out.get("L_cor")

        out["loss_vacl"] = L_cor


        # ---- attention maps (REQUIRED)
        if "X_v_att" not in out or "X_a_att" not in out:
            raise KeyError("VACLVA output must contain 'X_v_att' and 'X_a_att' for Stage-2 head.")




        # ============================================================
        # [KEPT] Strip heavy intermediates
        # ============================================================
        if self.cfg.strip_intermediates and not return_intermediates:
            for k in [
                "M_v", "M_a", "J_va", "J_av",
                "attn", "attn_weights",
                "pos_mask", "neg_mask",
                "logits", "logits_va", "logits_av",
                "intermediates",
            ]:
                out.pop(k, None)

        return out
