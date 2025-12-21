#!/usr/bin/env python3
# utils/Imp_indexing_help/probe_finetune_video_pt_lengths.py
# ============================================================
# Probe T_video from saved video .pt tensors in parallel.
#
# Input:  <batch_dir>/segment_paths_finetune.csv
# Output: <batch_dir>/segment_index_finetune.csv
# ============================================================

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def _probe_T_video_from_pt(v_pt: Path) -> Optional[int]:
    try:
        v = torch.load(v_pt, map_location="cpu")
        if not isinstance(v, torch.Tensor) or v.ndim != 4 or v.shape[0] != 3:
            return None
        return int(v.shape[1])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", type=str, required=True)
    ap.add_argument("--in-csv", type=str, default="segment_paths_finetune.csv")     # [CHANGED]
    ap.add_argument("--out-csv", type=str, default="segment_index_finetune.csv")    # [CHANGED]
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--fail-policy", choices=["skip", "zero", "error"], default="skip")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
    in_path = batch_dir / args.in_csv
    out_path = batch_dir / args.out_csv

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_path}")

    rows = []
    with in_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    t0 = time.time()
    total = len(rows)
    done = 0
    failed = 0

    def job(row: Dict[str, str]) -> Tuple[Dict[str, str], Optional[int]]:
        v_rel = row["video_rel"]
        v_pt = batch_dir / v_rel
        return row, _probe_T_video_from_pt(v_pt)

    results = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(job, row) for row in rows]
        for fut in as_completed(futs):
            row, T = fut.result()
            if T is None:
                failed += 1
                if args.fail_policy == "error":
                    raise RuntimeError(f"Failed to probe video pt: {row.get('video_rel')}")
                elif args.fail_policy == "zero":
                    T = 0
                else:  # skip
                    row = None  # type: ignore

            if row is not None:
                results.append((row, int(T)))  # type: ignore

            done += 1
            if done % int(args.log_every) == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                print(f"[probe_video_pt] done={done}/{total} failed={failed} rate={rate:.2f}/s elapsed={dt:.1f}s")

    results.sort(key=lambda x: (x[0]["clip_id"], int(x[0]["seg_idx"])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "seg_idx", "audio96_rel", "audio2048_rel", "video_rel", "T_video"])
        for row, T in results:
            w.writerow([
                row["clip_id"],
                int(row["seg_idx"]),
                row["audio96_rel"],
                row["audio2048_rel"],
                row["video_rel"],
                int(T),
            ])

    dt = time.time() - t0
    skipped = (failed if args.fail_policy == "skip" else 0)
    print(f"[probe_video_pt] wrote={len(results)} skipped={skipped} -> {out_path} elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
