# stage2_head.py
# =============================================================================
# Stage-2 classifier head (Lightning/DDP friendly) -- CLEAN (no fixed k)
# Uses BCEWithLogitsLoss with a SINGLE logit output (B,1).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class Stage2HeadConfig:
    num_classes: int = 1          # IMPORTANT: single logit for BCEWithLogitsLoss
    pool: str = "mean"            # "mean" or "max"
    use_layernorm: bool = False
    mlp_hidden: Optional[int] = None
    dropout: float = 0.0


class Stage2AVClassifierHead(nn.Module):
    """
    Consumes:
        X_v_att: (B, Dv, S)
        X_a_att: (B, Da, S)

    Produces:
        logits: (B, 1)    # single logit for BCEWithLogitsLoss
    """

    def __init__(self, d_v: int, d_a: int, cfg: Stage2HeadConfig = Stage2HeadConfig()):
        super().__init__()
        self.cfg = cfg

        in_dim = int(d_v + d_a)
        out_dim = int(cfg.num_classes)
        if out_dim != 1:
            raise ValueError(
                f"[Stage2AVClassifierHead] For BCEWithLogitsLoss, set cfg.num_classes=1, got {out_dim}."
            )

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
            raise ValueError(f"[Stage2AVClassifierHead] Expected (B,D,S), got {tuple(X.shape)}")

        if self.cfg.pool == "mean":
            return X.mean(dim=-1)            # (B, D)
        if self.cfg.pool == "max":
            return X.max(dim=-1).values      # (B, D)
        raise ValueError("[Stage2AVClassifierHead] cfg.pool must be 'mean' or 'max'")

    def forward(self, X_v_att: torch.Tensor, X_a_att: torch.Tensor) -> torch.Tensor:
        if X_v_att.shape[0] != X_a_att.shape[0]:
            raise ValueError("[Stage2AVClassifierHead] Batch size mismatch between modalities.")
        if X_v_att.ndim != 3 or X_a_att.ndim != 3:
            raise ValueError("[Stage2AVClassifierHead] Expected (B,D,S) for both modalities.")

        v = self._pool(X_v_att)              # (B, Dv)
        a = self._pool(X_a_att)              # (B, Da)
        z = torch.cat([v, a], dim=1)         # (B, Dv+Da)

        if self.ln is not None:
            z = self.ln(z)

        logits = self.head(z)                # (B, 1)
        return logits


# =============================================================================
# Small usage examples
# =============================================================================

@torch.no_grad()
def infer_probs_and_preds(
    head: Stage2AVClassifierHead,
    X_v_att: torch.Tensor,
    X_a_att: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Inference helper (no labels):
      - returns prob_fake in [0,1] and hard preds {0,1}
    """
    logits = head(X_v_att, X_a_att)                  # (B,1)
    prob_fake = torch.sigmoid(logits)                # (B,1)
    pred = (prob_fake >= threshold).long()           # (B,1)
    return {"logits": logits, "prob_fake": prob_fake, "pred": pred}


def eval_step_with_bce(
    head: Stage2AVClassifierHead,
    X_v_att: torch.Tensor,
    X_a_att: torch.Tensor,
    y: torch.Tensor,
    bce: Optional[nn.Module] = None,
) -> Dict[str, torch.Tensor]:
    """
    Evaluation step with labels:
      y can be (B,) or (B,1) with values in {0,1}
    """
    if bce is None:
        bce = nn.BCEWithLogitsLoss()

    logits = head(X_v_att, X_a_att)                  # (B,1)
    y = y.float().view(-1, 1)                        # (B,1)
    loss = bce(logits, y)

    prob_fake = torch.sigmoid(logits)                # (B,1)
    pred = (prob_fake >= 0.5).long()                 # (B,1)

    return {"loss": loss, "logits": logits, "prob_fake": prob_fake, "pred": pred}


# if __name__ == "__main__":
    # Dummy shapes:
    # B, Dv, Da, S = 4, 256, 256, 32
    # X_v_att = torch.randn(B, Dv, S)
    # X_a_att = torch.randn(B, Da, S)
    # y = torch.randint(0, 2, (B,))
    #
    # cfg = Stage2HeadConfig(pool="mean", use_layernorm=True, mlp_hidden=256, dropout=0.1)
    # head = Stage2AVClassifierHead(d_v=Dv, d_a=Da, cfg=cfg)
    #
    # # Inference
    # out_inf = infer_probs_and_preds(head, X_v_att, X_a_att)
    # print("prob_fake:", out_inf["prob_fake"].squeeze(-1))
    #
    # # Evaluation
    # out_eval = eval_step_with_bce(head, X_v_att, X_a_att, y)
    # print("loss:", float(out_eval["loss"]))
    # print("pred:", out_eval["pred"].squeeze(-1))
