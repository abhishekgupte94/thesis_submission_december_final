# wap_ensemble.py
# ===========================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ===========================================================
# [ADDED] Metrics imports (required for evaluators)
# ===========================================================
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def _safe_clamp_probs(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.clamp(p, eps, 1.0 - eps)


# ===========================================================
# [ADDED] Evaluators (AUCROC, ACC, F1) + "lumpy" metric fns
# ===========================================================

def _to_numpy_1d(x: torch.Tensor):
    return x.reshape(-1).detach().cpu().numpy()


def evaluate_probs(
    p: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate probabilistic predictions against ground truth.

    Args:
        p: probabilities in [0,1], shape (N,) or (N,1)
        y: labels in {0,1}, shape (N,) or (N,1)
        threshold: decision threshold for ACC and F1

    Returns:
        {"AUC": ..., "AUCROC": ..., "ACC": ..., "F1": ...}
    """
    p_np = _to_numpy_1d(p)
    y_np = _to_numpy_1d(y).astype(int)

    # AUC/ROC-AUC undefined if y has only one class
    if len(set(y_np.tolist())) < 2:
        aucroc = float("nan")
    else:
        aucroc = float(roc_auc_score(y_np, p_np))

    y_hat = (p_np >= threshold).astype(int)
    acc = float(accuracy_score(y_np, y_hat))
    f1 = float(f1_score(y_np, y_hat))

    return {
        "AUC": aucroc,      # kept for backward compatibility
        "AUCROC": aucroc,   # explicit alias
        "ACC": acc,
        "F1": f1,
    }


def evaluate_all(
    m_A: torch.Tensor,
    m_B: torch.Tensor,
    p_ens: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate Model A, Model B, and Ensemble together.
    """
    return {
        "Model_A": evaluate_probs(m_A, y, threshold),
        "Model_B": evaluate_probs(m_B, y, threshold),
        "Ensemble": evaluate_probs(p_ens, y, threshold),
    }


def metric_acc_at_threshold(threshold: float = 0.5) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """
    Gridsearch metric: ACC at fixed threshold.
    """
    def _fn(p: torch.Tensor, y: torch.Tensor) -> float:
        return float(evaluate_probs(p, y, threshold=threshold)["ACC"])
    return _fn


def metric_f1_at_threshold(threshold: float = 0.5) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """
    Gridsearch metric: F1 at fixed threshold.
    """
    def _fn(p: torch.Tensor, y: torch.Tensor) -> float:
        return float(evaluate_probs(p, y, threshold=threshold)["F1"])
    return _fn


def metric_aucroc() -> Callable[[torch.Tensor, torch.Tensor], float]:
    """
    Gridsearch metric: ROC-AUC (aka AUCROC). Higher is better.
    If undefined (single-class y), returns -inf so it won't be selected.
    """
    def _fn(p: torch.Tensor, y: torch.Tensor) -> float:
        auc = evaluate_probs(p, y)["AUCROC"]
        if auc != auc:  # NaN check
            return float("-inf")
        return float(auc)
    return _fn


# ===========================================================
# Core WAP (unchanged)
# ===========================================================

@dataclass
class WAPConfig:
    wB: float = 0.05          # default: heavily trust A
    clamp_eps: float = 1e-7   # safety clamp for numerical stability


class WAPEnsembler(nn.Module):
    """
    Weighted Average of Probabilities:
        p_ens = (1 - wB) * m_A + wB * m_B

    Inputs:
        m_A: (B,) or (B,1) probabilities for class=1 ("fake")
        m_B: (B,) or (B,1) probabilities for class=1 ("fake")

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

    def forward(self, m_A: torch.Tensor, m_B: torch.Tensor) -> torch.Tensor:
        m_A = _safe_clamp_probs(m_A, self.clamp_eps)
        m_B = _safe_clamp_probs(m_B, self.clamp_eps)
        wB = self._wB
        # broadcast-safe
        return (1.0 - wB) * m_A + wB * m_B


# ----------------------------
# Simple weight fitting utility
# ----------------------------

def fit_weight_gridsearch(
    m_A_val: torch.Tensor,
    m_B_val: torch.Tensor,
    y_val: torch.Tensor,
    metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    wB_grid: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Pick wB on a validation set by grid-search.

    Args:
        m_A_val, m_B_val: probabilities in [0,1], shape (N,) or (N,1)
        y_val: labels in {0,1}, shape (N,) or (N,1)
        metric_fn: function(score_probs, y) -> float (higher is better).
                   If None: uses ACC@0.5 via evaluator wrapper.
        wB_grid: tensor of candidate wB values. If None uses [0..0.5] fine grid.

    Returns:
        best_wB, stats dict
    """
    m_A_val = m_A_val.reshape(-1).detach().cpu()
    m_B_val = m_B_val.reshape(-1).detach().cpu()
    y_val = y_val.reshape(-1).detach().cpu().float()

    if wB_grid is None:
        # since A is strong, search mostly small wB; include a few bigger just in case
        wB_grid = torch.cat([
            torch.linspace(0.0, 0.10, 51),   # fine search near 0
            torch.linspace(0.12, 0.30, 10),
            torch.linspace(0.35, 0.50, 4),
        ]).unique(sorted=True)

    # [MODIFIED - minimal] default metric now routed through evaluator wrapper
    if metric_fn is None:
        metric_fn = metric_acc_at_threshold(threshold=0.5)

    best_wB = 0.0
    best_score = float("-inf")

    for wB in wB_grid:
        p = (1.0 - wB) * m_A_val + wB * m_B_val
        score = float(metric_fn(p, y_val))
        if score > best_score:
            best_score = score
            best_wB = float(wB.item())

    # also report baselines (same selection metric for apples-to-apples)
    score_A = float(metric_fn(m_A_val, y_val))
    score_B = float(metric_fn(m_B_val, y_val))
    score_ens = float(best_score)

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

class EnsembleModelWrapper(nn.Module):
    """
    Calls modelA + modelB, converts to probabilities, then WAP fuses.

    Assumptions:
      - Each model returns either:
          (a) logits shape (B,) or (B,1) for "fake"
          (b) probabilities already in [0,1] if you set `inputs_are_probs=True`
    """
    def __init__(
        self,
        npv_custom_model: nn.Module,
        lavdf_model: nn.Module,
        wap: WAPEnsembler,
        inputs_are_probs: bool = False,
    ):
        super().__init__()
        self.npv_custom_model = npv_custom_model
        self.lavdf_model = lavdf_model
        self.wap = wap
        self.inputs_are_probs = bool(inputs_are_probs)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        out_npv = self.npv_custom_model(batch)
        out_lavdf = self.lavdf_model(batch)

        # If models return dicts, adapt here:
        if isinstance(out_npv, dict):
            out_npv = out_npv["logits"]
        if isinstance(out_lavdf, dict):
            out_lavdf = out_lavdf["logits"]

        if self.inputs_are_probs:
            m_A = out_npv
            m_B = out_lavdf
        else:
            # binary logit -> probability
            m_A = torch.sigmoid(out_npv)
            m_B = torch.sigmoid(out_lavdf)

        p_ens = self.wap(m_A, m_B)
        return {"m_A": m_A, "m_B": m_B, "p_ens": p_ens}
