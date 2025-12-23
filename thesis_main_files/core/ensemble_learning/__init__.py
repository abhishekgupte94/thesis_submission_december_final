# wap_ensemble.py
# ============================================================
# [DROP-IN] Weighted Average of Probabilities (WAP) Ensembler
#
# Strategy:
#   p_ens = (1-wB) * pA + wB * pB
#   where wB is chosen on validation set to optimize a metric.
#
# Notes:
#   - Expects probabilities in [0,1] (apply sigmoid/softmax beforehand)
#   - No device moves, DDP/Lightning-friendly
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


def _safe_clamp_probs(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.clamp(p, eps, 1.0 - eps)


@dataclass
class WAPConfig:
    wB: float = 0.05          # default: heavily trust A
    clamp_eps: float = 1e-7   # safety clamp for numerical stability


class WAPEnsembler(nn.Module):
    """
    Weighted Average of Probabilities:
        p_ens = (1 - wB) * pA + wB * pB

    Inputs:
        pA: (B,) or (B,1) probabilities for class=1 ("fake")
        pB: (B,) or (B,1) probabilities for class=1 ("fake")

    Output:
        p_ens: same shape as inputs
    """
    def __init__(self, cfg: WAPConfig):
        super().__init__()
        # store as buffer so it moves with .to(device) if needed but isn't trainable
        self.register_buffer("_wB", torch.tensor(float(cfg.wB), dtype=torch.float32))
        self.clamp_eps = float(cfg.clamp_eps)

    @property
    def wB(self) -> float:
        return float(self._wB.item())

    def set_wB(self, wB: float) -> None:
        with torch.no_grad():
            self._wB.fill_(float(wB))

    def forward(self, pA: torch.Tensor, pB: torch.Tensor) -> torch.Tensor:
        pA = _safe_clamp_probs(pA, self.clamp_eps)
        pB = _safe_clamp_probs(pB, self.clamp_eps)
        wB = self._wB
        # broadcast-safe
        return (1.0 - wB) * pA + wB * pB


# ----------------------------
# Simple weight fitting utility
# ----------------------------

def fit_weight_gridsearch(
    pA_val: torch.Tensor,
    pB_val: torch.Tensor,
    y_val: torch.Tensor,
    metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    wB_grid: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Pick wB on a validation set by grid-search.

    Args:
        pA_val, pB_val: probabilities in [0,1], shape (N,) or (N,1)
        y_val: labels in {0,1}, shape (N,) or (N,1)
        metric_fn: function(score_probs, y) -> float (higher is better).
                   If None: uses a simple accuracy-at-0.5 metric.
        wB_grid: tensor of candidate wB values. If None uses [0..0.5] fine grid.

    Returns:
        best_wB, stats dict
    """
    pA_val = pA_val.reshape(-1).detach().cpu()
    pB_val = pB_val.reshape(-1).detach().cpu()
    y_val  = y_val.reshape(-1).detach().cpu().float()

    if wB_grid is None:
        # since A is strong, search mostly small wB; include a few bigger just in case
        wB_grid = torch.cat([
            torch.linspace(0.0, 0.10, 51),   # fine search near 0
            torch.linspace(0.12, 0.30, 10),
            torch.linspace(0.35, 0.50, 4),
        ]).unique(sorted=True)

    if metric_fn is None:
        def metric_fn(p: torch.Tensor, y: torch.Tensor) -> float:
            pred = (p >= 0.5).float()
            return float((pred == y).float().mean().item())

    best_wB = 0.0
    best_score = -1e18

    for wB in wB_grid:
        p = (1.0 - wB) * pA_val + wB * pB_val
        score = metric_fn(p, y_val)
        if score > best_score:
            best_score = score
            best_wB = float(wB.item())

    # also report baselines
    score_A = metric_fn(pA_val, y_val)
    score_B = metric_fn(pB_val, y_val)
    score_ens = best_score

    stats = {
        "score_A": float(score_A),
        "score_B": float(score_B),
        "score_ens_best": float(score_ens),
        "best_wB": float(best_wB),
    }
    return best_wB, stats


# ----------------------------
# Lightning-friendly wrapper
# ----------------------------

class TwoModelWAPWrapper(nn.Module):
    """
    Calls modelA + modelB, converts to probabilities, then WAP fuses.

    Assumptions:
      - Each model returns either:
          (a) logits shape (B,) or (B,1) for "fake"
          (b) probabilities already in [0,1] if you set `inputs_are_probs=True`
    """
    def __init__(
        self,
        modelA: nn.Module,
        modelB: nn.Module,
        wap: WAPEnsembler,
        inputs_are_probs: bool = False,
    ):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.wap = wap
        self.inputs_are_probs = bool(inputs_are_probs)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        outA = self.modelA(batch)
        outB = self.modelB(batch)

        # If models return dicts, adapt here:
        if isinstance(outA, dict): outA = outA["logits"]
        if isinstance(outB, dict): outB = outB["logits"]

        if self.inputs_are_probs:
            pA = outA
            pB = outB
        else:
            # binary logit -> probability
            pA = torch.sigmoid(outA)
            pB = torch.sigmoid(outB)

        p_ens = self.wap(pA, pB)
        return {"pA": pA, "pB": pB, "p_ens": p_ens}
