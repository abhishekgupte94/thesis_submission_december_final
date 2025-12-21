# ============================================================
# [PATCH] BatfdToVACLAdapter
#   - KEEP forward(...) unchanged
#   - [ADDED] adapt_and_run_vacl(...) which ACCEPTS vacl_wrapper
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# (imports for PRBExtraction etc remain as you already have)


class BatfdToVACLAdapter(nn.Module):
    """
    Converts PRBExtraction -> (X_v, X_a) for VACL.

    STRICT:
      - Does NOT instantiate VACLWrapper
      - Does NOT change VACLVA math
      - forward(...) remains pure adaptation
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # [KEPT] everything you already had: projections, MHA stubs, norms, etc.

    def forward(self, prb, *, target_S: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # ============================================================
        # [KEPT] YOUR EXISTING FORWARD IMPLEMENTATION EXACTLY
        #   - Select prb maps
        #   - project -> common
        #   - (optional) fusion
        #   - MHA stub
        #   - project -> (d_v,d_a)
        #   - match_len -> target_S
        # ============================================================
        raise NotImplementedError  # <-- remove: this is just a placeholder comment guard

    # ============================================================
    # [ADDED] Convenience: adapt PRB -> run VACLWrapper (passed in)
    # ============================================================
    @torch.no_grad()  # [ADDED] safe default; remove if you want grads through adapter+vacl
    def adapt_and_run_vacl(
        self,
        prb,
        *,
        vacl_wrapper: nn.Module,
        target_S: int,
        return_intermediates: Optional[bool] = None,
        **vacl_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        [ADDED] Convenience method.

        Inputs:
          prb: PRBExtraction (BATFD cut-point outputs)
          vacl_wrapper: an already-constructed VACLWrapper (caller owns instantiation)
          target_S: the S to force adapter outputs to
          return_intermediates: forwarded to vacl_wrapper if provided
          **vacl_kwargs: forwarded to vacl_wrapper (e.g., compute_infonce, etc.)

        Returns:
          dict: whatever vacl_wrapper returns (losses + X_v_att/X_a_att, etc.)
        """

        # [ADDED] 1) Adapt PRB -> VACL inputs
        X_v, X_a = self.forward(prb, target_S=int(target_S))

        # [ADDED] 2) Call the provided wrapper (NO instantiation here)
        if return_intermediates is None:
            vacl_out = vacl_wrapper(X_v=X_v, X_a=X_a, **vacl_kwargs)
        else:
            vacl_out = vacl_wrapper(
                X_v=X_v,
                X_a=X_a,
                return_intermediates=bool(return_intermediates),
                **vacl_kwargs,
            )

        return vacl_out


# prb = batfd_prb_extractor(video, audio)   # your existing PRB extraction
# vacl_out = adapter.adapt_and_run_vacl(
#     prb,
#     vacl_wrapper=vacl_wrapper,            # your VACLWrapper instance
#     target_S=desired_S,
#     return_intermediates=False,
# )
