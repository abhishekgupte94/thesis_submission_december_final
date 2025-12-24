import json
from typing import Dict, List, Tuple, Optional

import torch

# Import these from your wap_ensemble.py (same file is fine too)
from core.ensemble_learning.ensemble_metrics import  evaluate_all, metric_aucroc, metric_acc_at_threshold, metric_f1_at_threshold
from core.ensemble_learning.weighted_average_ensemble import  fit_weight_gridsearch

def _read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _key(clip_id: str, seg_idx: int) -> str:
    # stable key for dict joins
    return f"{clip_id}__{int(seg_idx)}"


def align_segmentA_with_segmentB(
    npv_rows: List[Dict],
    lavdf_rows: List[Dict],
    require_labels: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, int]]:
    """
    Align:
      - Model A (NPV): keys: clip_id, seg_idx, prob, (optional) label
      - Model B (LAVDF): keys: clip_id, seg_idx, prob_fake, (optional) label

    Join key:
      (clip_id, seg_idx)

    Returns:
      m_A: (N,) tensor of A probs
      m_B: (N,) tensor of B probs
      y:   (N,) tensor of labels (or None)
      stats: counts (matched, missing_in_A, missing_in_B, label_mismatch, etc.)
    """
    # Build maps
    a_map: Dict[str, Dict] = {}
    for r in npv_rows:
        cid = str(r["clip_id"])
        sidx = int(r["seg_idx"])
        a_map[_key(cid, sidx)] = {
            "prob": float(r["prob"]),
            "label": r.get("label", None),
        }

    b_map: Dict[str, Dict] = {}
    for r in lavdf_rows:
        cid = str(r["clip_id"])
        sidx = int(r["seg_idx"])
        b_map[_key(cid, sidx)] = {
            "prob_fake": float(r["prob_fake"]),
            "label": r.get("label", None),
        }

    keys_a = set(a_map.keys())
    keys_b = set(b_map.keys())
    common_keys = sorted(list(keys_a.intersection(keys_b)))

    missing_in_A = len(keys_b - keys_a)
    missing_in_B = len(keys_a - keys_b)

    mA_list: List[float] = []
    mB_list: List[float] = []
    y_list: List[int] = []

    label_mismatch = 0
    missing_label = 0

    for k in common_keys:
        prob_A = a_map[k]["prob"]
        prob_B = b_map[k]["prob_fake"]

        # label handling
        yA = a_map[k].get("label", None)
        yB = b_map[k].get("label", None)

        if require_labels:
            if yA is None and yB is None:
                missing_label += 1
                continue

            # Prefer label from A if present else from B
            y = int(yA) if yA is not None else int(yB)

            if yA is not None and yB is not None and int(yA) != int(yB):
                label_mismatch += 1
                # keep A's label by default
                y = int(yA)

            y_list.append(y)

        mA_list.append(prob_A)
        mB_list.append(prob_B)

    m_A = torch.tensor(mA_list, dtype=torch.float32)
    m_B = torch.tensor(mB_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.int64) if require_labels else None

    stats = {
        "rows_A": len(npv_rows),
        "rows_B": len(lavdf_rows),
        "unique_pairs_A": len(keys_a),
        "unique_pairs_B": len(keys_b),
        "matched_pairs": len(common_keys),
        "used_pairs_after_label_filter": int(m_A.numel()),
        "missing_in_A": int(missing_in_A),
        "missing_in_B": int(missing_in_B),
        "missing_label_dropped": int(missing_label),
        "label_mismatch": int(label_mismatch),
    }
    return m_A, m_B, y, stats


def run_wap_offline_from_jsonl(
    npv_jsonl_path: str,
    lavdf_jsonl_path: str,
    threshold: float = 0.5,
    selection: str = "aucroc",  # "aucroc" | "acc" | "f1"
):
    npv_rows = _read_jsonl(npv_jsonl_path)
    lavdf_rows = _read_jsonl(lavdf_jsonl_path)

    m_A, m_B, y, align_stats = align_segmentA_with_segmentB(npv_rows, lavdf_rows, require_labels=True)
    print("ALIGN STATS:", align_stats)

    # Choose gridsearch selection metric
    if selection.lower() == "aucroc":
        metric_fn = metric_aucroc()
    elif selection.lower() == "acc":
        metric_fn = metric_acc_at_threshold(threshold=threshold)
    elif selection.lower() == "f1":
        metric_fn = metric_f1_at_threshold(threshold=threshold)
    else:
        raise ValueError(f"Unknown selection metric: {selection}")

    best_wB, gs_stats = fit_weight_gridsearch(m_A, m_B, y, metric_fn=metric_fn)
    print("GRIDSEARCH STATS:", gs_stats)

    p_ens = (1.0 - best_wB) * m_A + best_wB * m_B
    report = evaluate_all(m_A, m_B, p_ens, y, threshold=threshold)

    print("REPORT:")
    for name, scores in report.items():
        print(name, scores)

    return {
        "best_wB": best_wB,
        "align_stats": align_stats,
        "gridsearch_stats": gs_stats,
        "report": report,
    }


# Example usage:
# results = run_wap_offline_from_jsonl(
#     npv_jsonl_path="npv_preds.jsonl",
#     lavdf_jsonl_path="preds.jsonl",
#     threshold=0.5,
#     selection="aucroc",
# )
