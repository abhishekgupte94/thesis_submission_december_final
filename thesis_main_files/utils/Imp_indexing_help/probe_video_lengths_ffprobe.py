#!/usr/bin/env python3
# probe_video_lengths_ffprobe.py
#
# Reads a paths manifest and computes T_video (frame count) with ffprobe in parallel.
# Writes an enriched CSV that your build_index can load quickly.

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple


def probe_num_frames_ffprobe(mp4_path: Path) -> Optional[int]:
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames,nb_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(mp4_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip().splitlines()
        for line in reversed(out):
            line = line.strip()
            if line.isdigit():
                n = int(line)
                if n > 0:
                    return n
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", type=str, required=True, help="Full path to <offline_root>/<batch_name>")
    ap.add_argument("--in-csv", type=str, default="segment_paths.csv")
    ap.add_argument("--out-csv", type=str, default="segment_index.csv")
    ap.add_argument("--workers", type=int, default=16, help="Thread workers (ffprobe is external; threads are fine).")
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--fail-policy", choices=["skip", "zero", "error"], default="skip")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
    in_path = batch_dir / args.in_csv
    out_path = batch_dir / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
        v_mp4 = batch_dir / v_rel
        return row, probe_num_frames_ffprobe(v_mp4)

    results = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(job, row) for row in rows]
        for fut in as_completed(futs):
            row, T = fut.result()
            if T is None:
                failed += 1
                if args.fail_policy == "error":
                    raise RuntimeError(f"ffprobe failed for video_rel={row.get('video_rel')}")
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
                print(f"[ffprobe] done={done}/{total} failed={failed} rate={rate:.2f}/s elapsed={dt:.1f}s")

    # Sort deterministically (matches your dataset ordering)
    results.sort(key=lambda x: (x[0]["clip_id"], int(x[0]["seg_idx"])))

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "seg_idx", "audio_rel", "video_rel", "T_video"])
        for row, T in results:
            w.writerow([row["clip_id"], int(row["seg_idx"]), row["audio_rel"], row["video_rel"], int(T)])

    dt = time.time() - t0
    print(f"[ffprobe] wrote={len(results)} skipped={failed if args.fail_policy=='skip' else 0} -> {out_path} elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
