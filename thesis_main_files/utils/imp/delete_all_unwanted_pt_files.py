#!/usr/bin/env python3
"""
delete_pt_files_safe.py

Safely and efficiently delete ONLY `.pt` files under a directory.
Explicitly protects `.mp4` files (never deleted).

Guarantees:
- Only files ending exactly with `.pt` are deleted
- `.mp4` files are explicitly rejected
- Optional path safety guard via --must-contain
- Clean progress + summary logging
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class DeleteStats:
    scanned: int = 0
    pt_matched: int = 0
    deleted: int = 0
    failed: int = 0
    mp4_skipped: int = 0
    bytes_deleted: int = 0


def iter_files(root: Path) -> Iterable[Path]:
    # Walk everything once; filter manually for safety
    yield from root.rglob("*")


def fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"


def main() -> int:
    ap = argparse.ArgumentParser(description="Safely delete ONLY .pt files (mp4 protected).")
    ap.add_argument("--root", required=True, help="Root directory (recursive).")
    ap.add_argument("--dry-run", action="store_true", help="List deletions without deleting.")
    ap.add_argument("--log-every", type=int, default=2000, help="Progress log frequency.")
    ap.add_argument("--print-each", action="store_true", help="Print every deleted file.")
    ap.add_argument(
        "--must-contain",
        type=str,
        default="",
        help="Safety guard: root path must contain this substring.",
    )
    args = ap.parse_args()

    t0 = time.time()
    root = Path(args.root).expanduser().resolve()

    if not root.is_dir():
        print(f"[delete_pt][error] Not a directory: {root}", file=sys.stderr)
        return 2

    if args.must_contain and args.must_contain not in str(root):
        print(
            f"[delete_pt][safety] Refusing to run: root path does not contain "
            f"'{args.must_contain}'",
            file=sys.stderr,
        )
        return 3

    stats = DeleteStats()

    print(f"[delete_pt] root={root}")
    print(f"[delete_pt] dry_run={args.dry_run} log_every={args.log_every}")
    if args.must_contain:
        print(f"[delete_pt] safety_guard: must_contain='{args.must_contain}'")

    for p in iter_files(root):
        stats.scanned += 1

        try:
            if not p.is_file():
                continue
        except OSError:
            stats.failed += 1
            continue

        suffix = p.suffix.lower()

        # ---- HARD SAFETY RULES ----
        if suffix == ".mp4":
            stats.mp4_skipped += 1
            continue

        if suffix != ".pt":
            continue
        # ---------------------------

        stats.pt_matched += 1

        size = 0
        try:
            size = p.stat().st_size
        except OSError:
            pass

        if args.dry_run:
            if args.print_each:
                print(f"[dryrun] {p}")
            continue

        try:
            p.unlink()
            stats.deleted += 1
            stats.bytes_deleted += size

            if args.print_each:
                print(f"[deleted] {p}")

            if stats.deleted % max(1, args.log_every) == 0:
                elapsed = time.time() - t0
                rate = stats.deleted / elapsed if elapsed > 0 else 0.0
                print(
                    f"[delete_pt][progress] deleted={stats.deleted} "
                    f"pt_matched={stats.pt_matched} scanned={stats.scanned} "
                    f"mp4_skipped={stats.mp4_skipped} failed={stats.failed} "
                    f"elapsed_sec={elapsed:.1f} rate={rate:.2f} files/s "
                    f"bytes_deleted={fmt_bytes(stats.bytes_deleted)}"
                )
        except OSError as e:
            stats.failed += 1
            print(f"[delete_pt][warn] failed to delete {p} err={e}", file=sys.stderr)

    elapsed = time.time() - t0
    rate = stats.deleted / elapsed if elapsed > 0 else 0.0

    tag = "dryrun" if args.dry_run else "done"
    print(
        f"[delete_pt][{tag}] scanned={stats.scanned} pt_matched={stats.pt_matched} "
        f"deleted={stats.deleted} mp4_skipped={stats.mp4_skipped} "
        f"failed={stats.failed} elapsed_sec={elapsed:.1f} "
        f"rate={rate:.2f} files/s bytes_deleted={fmt_bytes(stats.bytes_deleted)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
