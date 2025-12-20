# stage2_head.py
# =============================================================================
# Stage-2 classifier head (Lightning/DDP friendly)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class Stage2HeadConfig:
    num_classes: int = 2
    pool: str = "mean"           # "mean" or "max"
    use_layernorm: bool = False
    mlp_hidden: Optional[int] = None
    dropout: float = 0.0


class Stage2AVClassifierHead(nn.Module):
    """
    Consumes:
        X_v_att: (B, k, S)
        X_a_att: (B, k, S)

    Produces:
        logits: (B, C)
        P:      (B, C)
    """

    def __init__(self, k: int, cfg: Stage2HeadConfig = Stage2HeadConfig()):
        super().__init__()
        self.k = int(k)
        self.cfg = cfg

        in_dim = 2 * self.k
        out_dim = int(cfg.num_classes)

        self.ln = nn.LayerNorm(in_dim) if cfg.use_layernorm else None

        if cfg.mlp_hidden is None:
            self.head = nn.Linear(in_dim, out_dim)
        else:
            hid = int(cfg.mlp_hidden)
            self.head = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ReLU(inplace=True),
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(hid, out_dim),
            )

    def _pool(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3:
            raise ValueError(f"[Stage2AVClassifierHead] Expected (B,k,S), got {tuple(X.shape)}")
        if X.shape[1] != self.k:
            raise ValueError(f"[Stage2AVClassifierHead] Expected channel dim k={self.k}, got {X.shape[1]}")

        if self.cfg.pool == "mean":
            return X.mean(dim=-1)
        if self.cfg.pool == "max":
            return X.max(dim=-1).values
        raise ValueError("cfg.pool must be 'mean' or 'max'")

    def forward(self, X_v_att: torch.Tensor, X_a_att: torch.Tensor) -> Dict[str, torch.Tensor]:
        if X_v_att.shape[0] != X_a_att.shape[0]:
            raise ValueError("[Stage2AVClassifierHead] Batch size mismatch between modalities.")

        v = self._pool(X_v_att)  # (B, k)
        a = self._pool(X_a_att)  # (B, k)
        z = torch.cat([v, a], dim=1)  # (B, 2k)

        if self.ln is not None:
            z = self.ln(z)

        logits = self.head(z)  # (B, C)
        P = torch.softmax(logits, dim=-1)

        return {"logits": logits, "P": P}


def bce_paper_eq18(P: torch.Tensor, y_onehot: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    L_ce = -(1/N) * sum_i [ y_i*log(P_i) + (1-y_i)*log(1-P_i) ]
    """
    if P.shape != y_onehot.shape:
        raise ValueError(f"[bce_paper_eq18] Shape mismatch: P{tuple(P.shape)} vs y{tuple(y_onehot.shape)}")

    P = P.clamp(eps, 1.0 - eps)
    y = y_onehot.to(dtype=P.dtype)

    loss_elem = -(y * torch.log(P) + (1.0 - y) * torch.log(1.0 - P))
    loss_per_sample = loss_elem.view(loss_elem.shape[0], -1).sum(dim=1)
    return loss_per_sample.mean()

