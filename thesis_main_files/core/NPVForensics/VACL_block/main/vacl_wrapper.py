import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.NPVForensics.VACL_block.vacl_block import VACLVA


# ======================================================================
# [NEW] VACL projection-head wrapper for integration as ModuleB / ModuleC
# ======================================================================

class VACLProjectionHead(nn.Module):
    """
    Wrapper around VACLVA that makes it behave like a projection head
    suitable for use as a sub-architecture in a larger model (e.g. ModuleB
    or ModuleC inside RandomlyNamedSA).

    Responsibilities
    ----------------
    1. Accept modality-specific features for V and A.
    2. Run the VACLVA block to obtain fused sequence features X_va and the
       VA correlation loss.
    3. Pool X_va over time to get a per-video / per-sample embedding.
    4. Project that embedding to an output dimension expected by the main
       architecture (e.g. d_b).

    Intended usage
    --------------
    - Input features may be in either:
        * "bsd": (B, S, D)  [time-major, common in transformer stacks]
        * "bds": (B, D, S)  [channel-first, what VACLVA expects]
      You specify this via `input_layout`.

    - The forward method returns a dict so that the main LightningModule
      can:
        * use `out["proj"]` as the projection output to feed into
          downstream modules (e.g. ModuleC).
        * add `lambda_cor * out["L_cor"]` as an auxiliary loss term.

    Shapes
    ------
    X_v_in : (B, S, d_v) or (B, d_v, S) depending on `input_layout`
    X_a_in : (B, S, d_a) or (B, d_a, S)

    proj   : (B, out_dim)               # pooled + projected embedding
    """

    def __init__(
        self,
        d_v: int,
        d_a: int,
        seq_len: int,
        k: int,
        out_dim: int,
        mu: float = 0.5,
        input_layout: str = "bsd",
        pool: str = "mean",
    ):
        """
        Args
        ----
        d_v, d_a  : feature dims for V and A features.
        seq_len   : S, maximum sequence length.
        k         : hidden dimension used inside VACLVA.
        out_dim   : output dimension of the projection head (e.g. d_b).
        mu        : μ term for correlation loss in VACLVA.
        input_layout:
            "bsd" -> expect (B, S, D) and internally transpose to (B, D, S).
            "bds" -> expect (B, D, S) and pass directly to VACLVA.
        pool:
            "mean" -> temporal mean over S
            "max"  -> temporal max over S
            "cls"  -> take X_va[:,:,0] as sequence summary
        """
        super().__init__()
        assert input_layout in ("bsd", "bds"), \
            f"Unsupported input_layout={input_layout}"
        assert pool in ("mean", "max", "cls"), \
            f"Unsupported pool={pool}"

        self.input_layout = input_layout
        self.pool = pool

        # Core VACL block (unchanged)
        self.vacl = VACLVA(
            d_v=d_v,
            d_a=d_a,
            seq_len=seq_len,
            k=k,
            mu=mu,
        )

        # Simple linear projection from concatenated (V,A) feature dim
        # (d_v + d_a) → out_dim
        self.proj = nn.Linear(d_v + d_a, out_dim)

    def _to_bds(self, X: torch.Tensor) -> torch.Tensor:
        """
        Convert input to (B, D, S) layout expected by VACLVA if needed.
        """
        if self.input_layout == "bds":
            return X
        # "bsd": (B, S, D) -> (B, D, S)
        return X.transpose(1, 2)

    def _pool_temporal(self, X_va: torch.Tensor) -> torch.Tensor:
        """
        X_va : (B, D, S) -> (B, D) by temporal pooling.
        """
        if self.pool == "mean":
            return X_va.mean(dim=-1)
        if self.pool == "max":
            return X_va.max(dim=-1).values
        # "cls" token: first time-step
        return X_va[..., 0]

    def forward(
        self,
        X_v: torch.Tensor,
        X_a: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        Forward pass through VACL + projection head.

        Args
        ----
        X_v, X_a : input features for V and A.
                   Shape depends on `input_layout`.

        return_dict:
            - If True:  return a dict with projection + all VACL outputs.
            - If False: return only the projected embedding (B, out_dim).

        Returns
        -------
        When return_dict=True:
            {
              "proj"    : (B, out_dim),    # main projection head output
              "z"       : (B, d_v + d_a),  # pooled X_va before projection
              "X_va"    : (B, d_v + d_a, S),
              "L_cor"   : scalar correlation loss,
              "Loss_va" : same as L_cor,
              ...       # all other keys from VACLVA
            }

        When return_dict=False:
            proj : (B, out_dim)
        """
        # Ensure layout is (B, D, S) for the VACL block
        X_v_bds = self._to_bds(X_v)
        X_a_bds = self._to_bds(X_a)

        vacl_out = self.vacl(X_v_bds, X_a_bds)  # core VACL run
        X_va = vacl_out["X_va"]                 # (B, d_v + d_a, S)

        # Temporal pooling + projection
        z = self._pool_temporal(X_va)           # (B, d_v + d_a)
        proj = self.proj(z)                     # (B, out_dim)

        if not return_dict:
            return proj

        # Merge everything into a single dict for the main architecture
        out = dict(vacl_out)
        out["z"] = z
        out["proj"] = proj
        return out
