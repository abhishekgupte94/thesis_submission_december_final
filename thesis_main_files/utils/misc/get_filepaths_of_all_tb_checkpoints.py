#!/usr/bin/env python3
"""
utils/get_filepaths_of_all_tb_checkpoints.py

Find Lightning "last.ckpt" for specified TensorBoardLogger run names.

Typical layout:
  tb_logs/<run_name>/version_<N>/checkpoints/last.ckpt

Selection modes:
  - last-version (DEFAULT): pick the highest version_<N>
  - newest-mtime: pick the last.ckpt with newest modified time

Usage:
  # from thesis_main_files
  PYTHONPATH="$PWD" python utils/get_filepaths_of_all_tb_checkpoints.py --runs stage1_ssl stage2_finetune_main

  # choose selection mode
  PYTHONPATH="$PWD" python utils/get_filepaths_of_all_tb_checkpoints.py --runs stage1_ssl --pick newest-mtime

  # print rsync commands too
  PYTHONPATH="$PWD" python utils/get_filepaths_of_all_tb_checkpoints.py --runs stage1_ssl --print-rsync
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_VERSION_RE = re.compile(r"^version_(\d+)$")


def _latest_mtime(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _pick_last_version_ckpt(run_dir: Path) -> Tuple[Optional[Path], List[Path]]:
    """
    Picks ckpt from the highest version_<N>.
    Returns (chosen_ckpt, all_last_ckpts_found).
    """
    # Find version dirs
    version_dirs: List[Tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = _VERSION_RE.match(p.name)
        if not m:
            continue
        version_dirs.append((int(m.group(1)), p))

    if not version_dirs:
        # fallback: find any last.ckpt under run_dir
        all_last = sorted(run_dir.rglob("checkpoints/last.ckpt"))
        return _latest_mtime(all_last), all_last

    # Highest version number
    version_dirs.sort(key=lambda t: t[0])
    last_ver_num, last_ver_dir = version_dirs[-1]

    ckpt = last_ver_dir / "checkpoints" / "last.ckpt"
    if ckpt.exists():
        # also collect all matches for printing
        all_last = sorted(run_dir.glob("version_*/checkpoints/last.ckpt"))
        if not all_last:
            all_last = [ckpt]
        return ckpt, all_last

    # If that version exists but doesn't have last.ckpt, fallback
    all_last = sorted(run_dir.glob("version_*/checkpoints/last.ckpt"))
    if not all_last:
        all_last = sorted(run_dir.rglob("checkpoints/last.ckpt"))
    return _latest_mtime(all_last), all_last


def find_last_ckpt_for_run(tb_logs: Path, run_name: str, pick: str) -> Tuple[Optional[Path], List[Path]]:
    """
    Returns:
      (chosen_last_ckpt, all_last_ckpts_found)
    """
    run_dir = tb_logs / run_name
    if not run_dir.exists():
        return None, []

    if pick == "last-version":
        return _pick_last_version_ckpt(run_dir)

    # pick == "newest-mtime"
    all_last = sorted(run_dir.glob("version_*/checkpoints/last.ckpt"))
    if not all_last:
        all_last = sorted(run_dir.rglob("checkpoints/last.ckpt"))
    chosen = _latest_mtime(all_last)
    return chosen, all_last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb-logs", type=str, default="tb_logs", help="Path to tb_logs directory (relative or absolute).")
    ap.add_argument("--runs", nargs="+", required=True, help="Run names, e.g. stage1_ssl stage2_finetune_main")
    ap.add_argument(
        "--pick",
        choices=["last-version", "newest-mtime"],
        default="last-version",
        help="How to choose which last.ckpt to return per run.",
    )
    ap.add_argument("--print-all", action="store_true", help="Print all matching last.ckpt files per run.")
    ap.add_argument("--print-rsync", action="store_true", help="Also print rsync commands to download chosen ckpts.")
    args = ap.parse_args()

    tb_logs = Path(args.tb_logs).expanduser().resolve()
    if not tb_logs.exists():
        raise SystemExit(f"[ERROR] tb_logs path does not exist: {tb_logs}")

    results: Dict[str, Optional[Path]] = {}

    print(f"[INFO] tb_logs: {tb_logs}")
    print(f"[INFO] pick mode: {args.pick}")

    for run in args.runs:
        chosen, all_last = find_last_ckpt_for_run(tb_logs, run, pick=args.pick)
        results[run] = chosen

        if chosen is None:
            print(f"\n[RUN] {run}\n  -> NOT FOUND")
            continue

        print(f"\n[RUN] {run}\n  -> chosen: {chosen}")

        if args.print_all and all_last:
            print("  -> all matches:")
            for p in all_last:
                print(f"     - {p}")

    print("\n[ABS_PATHS]")
    for run, p in results.items():
        print(f"{run}: {str(p) if p else ''}")

    if args.print_rsync:
        print("\n[RSYNC_COMMANDS]  (run on your LOCAL machine; replace <LAMBDA_IP>)")
        for run, p in results.items():
            if not p:
                continue
            print(f'rsync -avP ubuntu@<LAMBDA_IP>:"{p}" ~/Downloads/{run}_last.ckpt')


if __name__ == "__main__":
    main()
