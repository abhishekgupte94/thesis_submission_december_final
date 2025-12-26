#!/usr/bin/env python3
"""
utils/find_last_ckpts.py

Find Lightning "last.ckpt" for specified TensorBoardLogger run names.

Typical layout:
  tb_logs/<run_name>/version_<N>/checkpoints/last.ckpt

Usage:
  # from thesis_main_files
  PYTHONPATH="$PWD" python utils/find_last_ckpts.py --runs stage1_ssl stage2_finetune_main

  # with custom tb_logs root
  PYTHONPATH="$PWD" python utils/find_last_ckpts.py --tb-logs tb_logs --runs stage1_ssl

  # print only newest per run (default), plus rsync commands
  PYTHONPATH="$PWD" python utils/find_last_ckpts.py --runs stage1_ssl --print-rsync
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _latest_mtime(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def find_last_ckpt_for_run(tb_logs: Path, run_name: str) -> Tuple[Optional[Path], List[Path]]:
    """
    Returns:
      (best_guess_last_ckpt, all_last_ckpts_found)
    """
    run_dir = tb_logs / run_name
    if not run_dir.exists():
        return None, []

    # Look for tb_logs/<run>/version_*/checkpoints/last.ckpt
    all_last = sorted(run_dir.glob("version_*/checkpoints/last.ckpt"))

    # Some people use "lightning_logs" naming or nested structures; as a fallback:
    if not all_last:
        all_last = sorted(run_dir.rglob("checkpoints/last.ckpt"))

    chosen = _latest_mtime(all_last)
    return chosen, all_last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb-logs", type=str, default="tb_logs", help="Path to tb_logs directory (relative or absolute).")
    ap.add_argument("--runs", nargs="+", required=True, help="Run names, e.g. stage1_ssl stage2_finetune_main")
    ap.add_argument("--print-all", action="store_true", help="Print all matching last.ckpt files per run.")
    ap.add_argument("--print-rsync", action="store_true", help="Also print rsync commands to download chosen ckpts.")
    args = ap.parse_args()

    tb_logs = Path(args.tb_logs).expanduser().resolve()
    if not tb_logs.exists():
        raise SystemExit(f"[ERROR] tb_logs path does not exist: {tb_logs}")

    results: Dict[str, Optional[Path]] = {}

    print(f"[INFO] tb_logs: {tb_logs}")
    for run in args.runs:
        chosen, all_last = find_last_ckpt_for_run(tb_logs, run)
        results[run] = chosen

        if chosen is None:
            print(f"\n[RUN] {run}\n  -> NOT FOUND")
            continue

        print(f"\n[RUN] {run}\n  -> chosen: {chosen}")

        if args.print_all and all_last:
            print("  -> all matches:")
            for p in all_last:
                print(f"     - {p}")

    # Machine-readable section (easy to copy/paste)
    print("\n[ABS_PATHS]")
    for run, p in results.items():
        print(f"{run}: {str(p) if p else ''}")

    if args.print_rsync:
        print("\n[RSYNC_COMMANDS]  (run on your LOCAL machine; replace <LAMBDA_IP>)")
        for run, p in results.items():
            if not p:
                continue
            # Build remote path by stripping local resolve assumption:
            # If you're running this on Lambda, chosen is already the correct absolute path to use remotely.
            print(f'rsync -avP ubuntu@<LAMBDA_IP>:"{p}" ~/Downloads/{run}_last.ckpt')


if __name__ == "__main__":
    main()
