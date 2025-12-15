from typing import Dict, Union

import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# [DDP / LIGHTNING FRIENDLY IMPORT]
# ----------------------------------------------------------------------
# Avoid hard-coded single import path so this works on:
#   - Mac sanity scripts
#   - A100 training box
#   - different repo roots
# ----------------------------------------------------------------------
try:
    from core.NPVForensics.VACL_block.vacl_block import VACLVA  # type: ignore
except Exception:  # pragma: no cover
    from vacl_block import VACLVA


class VACLProjectionHead(nn.Module):
    """
    Projection-head wrapper around VACLVA (sub-architecture).

    Responsibilities:
      1) Accept V/A features from upstream (post-SWIN).
      2) Convert layout to (B, D, S) for VACLVA.
      3) Run VACLVA (Eq. 9–16 unchanged).
      4) Pool X_va over S to get a per-sample vector.
      5) Linear projection to out_dim.

    input_layout supported:
      - "bsd": (B, S, D)
      - "bds": (B, D, S)
      - "sd" : (S, D)   (unbatched)
      - "ds" : (D, S)   (unbatched)

    pool:
      - "mean", "max", "cls"
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
        force_fp32_inputs: bool = True,
        return_intermediates: bool = False,
    ):
        super().__init__()
        assert input_layout in ("bsd", "bds", "sd", "ds"), f"Unsupported input_layout={input_layout}"
        assert pool in ("mean", "max", "cls"), f"Unsupported pool={pool}"

        # [CONFIG]
        self.input_layout = input_layout
        self.pool = pool
        # No-AMP friendly: cast incoming features to fp32 before VACL math.
        self.force_fp32_inputs = bool(force_fp32_inputs)
        # DDP efficiency: intermediates include large SxS maps; keep off by default.
        self.return_intermediates = bool(return_intermediates)

        # Core VACL block (Eq. 9–16 unchanged)
        self.vacl = VACLVA(d_v=d_v, d_a=d_a, seq_len=seq_len, k=k, mu=mu)

        # Projection head: (d_v + d_a) -> out_dim
        self.proj = nn.Linear(d_v + d_a, out_dim)

    def _to_bds(self, X: torch.Tensor) -> torch.Tensor:
        """
        Convert input to (B, D, S) for VACLVA.
        """
        if self.input_layout == "bds":
            if X.dim() != 3:
                raise ValueError(f"Expected (B, D, S) for 'bds', got {tuple(X.shape)}")
            return X.contiguous()

        if self.input_layout == "bsd":
            if X.dim() != 3:
                raise ValueError(f"Expected (B, S, D) for 'bsd', got {tuple(X.shape)}")
            return X.transpose(1, 2).contiguous()

        if self.input_layout == "sd":
            if X.dim() != 2:
                raise ValueError(f"Expected (S, D) for 'sd', got {tuple(X.shape)}")
            return X.transpose(0, 1).unsqueeze(0).contiguous()  # (1, D, S)

        if self.input_layout == "ds":
            if X.dim() != 2:
                raise ValueError(f"Expected (D, S) for 'ds', got {tuple(X.shape)}")
            return X.unsqueeze(0).contiguous()  # (1, D, S)

        raise ValueError(f"Unsupported input_layout={self.input_layout}")

    def _pool_temporal(self, X_va: torch.Tensor) -> torch.Tensor:
        """
        X_va : (B, D, S) -> (B, D)
        """
        if self.pool == "mean":
            return X_va.mean(dim=-1)
        if self.pool == "max":
            return X_va.max(dim=-1).values
        return X_va[..., 0]  # "cls"

    def forward(
        self,
        X_v: torch.Tensor,
        X_a: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
          - if return_dict: dict containing proj + L_cor (+ optional intermediates)
          - else: proj only
        """
        X_v_bds = self._to_bds(X_v)
        X_a_bds = self._to_bds(X_a)

        if self.force_fp32_inputs:
            if X_v_bds.dtype != torch.float32:
                X_v_bds = X_v_bds.float()
            if X_a_bds.dtype != torch.float32:
                X_a_bds = X_a_bds.float()

        vacl_out = self.vacl(
            X_v_bds,
            X_a_bds,
            return_intermediates=self.return_intermediates,
        )

        X_va = vacl_out["X_va"]  # (B, d_v + d_a, S)

        z = self._pool_temporal(X_va)  # (B, d_v + d_a)
        proj = self.proj(z)            # (B, out_dim)

        if not return_dict:
            return proj

        out = dict(vacl_out)
        out["z"] = z
        out["proj"] = proj
        return out
