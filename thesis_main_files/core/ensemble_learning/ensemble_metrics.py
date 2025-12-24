# ===========================================================
# [DROP-IN][EVAL] Lumpy metric fns + evaluators (AUCROC, ACC, F1)
# Copy-paste into your wap_ensemble.py (near the top, after imports)
# ===========================================================
from __future__ import annotations
from typing import Callable
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch


import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

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


# ===========================================================
# [DROP-IN][METRICS] "lumpy" metric functions for grid-search
# (each returns a callable metric_fn(p, y)->float)
# ===========================================================

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
