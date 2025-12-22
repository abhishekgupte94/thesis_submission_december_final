#!/usr/bin/env python3
# ============================================================
# [SCRIPT][PATCHED v2] sample_and_move_videos_then_make_timestamps_json.py
#
# HARD REQUIREMENT (your ask):
#   - Match ONLY rows where CSV column `file` starts with "test/"
#
# Stage 1:
#   - Filter rows: file.startswith("test/")
#   - Sample from those rows: 80% fake (label=1), 20% real (label=0)
#   - MOVE ONLY files that are actually FOUND on disk
#
# Stage 2:
#   - AFTER MOVING, create JSON ONLY from the rows whose videos were
#     successfully moved (matched files)
#   - JSON format:
#       { "000001.mp4": [[start,end], ...], ... }
#     Key is the video filename WITH extension.
#
# Notes:
#   - Robust src resolution handles case where src_root already points to ".../test"
#     while CSV contains "test/xxx.mp4"
#   - Destination preserves CSV subpath by default (so dst_root/test/xxx.mp4)
#   - --dry-run does NOTHING to disk
# ============================================================

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# -----------------------------
# [CONFIG]
# -----------------------------
@dataclass
class SampleMoveConfig:
    csv_path: Path

    # Videos are located under src_video_root (we resolve CSV `file` robustly)
    src_video_root: Path

    # Moved videos go under dst_video_root, preserving the CSV relative path by default
    dst_video_root: Path

    # JSON output directory
    json_out_dir: Path
    json_filename: str = "timestamps_without_words_sampled.json"

    # Sampling controls
    total_samples: Optional[int] = None
    fake_ratio: float = 0.80
    real_ratio: float = 0.20
    seed: int = 1337

    # STRICT prefix filter (ONLY accept rows whose `file` begins with this)
    required_file_prefix: str = "test/"

    # Behavior
    dry_run: bool = False
    overwrite: bool = False

    # Optional: used for robust src lookup (strip this prefix if src root already points to it)
    strip_prefix_for_src: Optional[str] = "test"

    # Logging
    log_every: int = 500


# -----------------------------
# [HELPERS]
# -----------------------------
def _as_int_label(v: str) -> int:
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"Could not parse label={v!r} as int") from e


def _safe_parse_timestamps_wo_words(s: str) -> List[List[float]]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Failed to parse timestamps_without_words={s[:160]!r}...") from e

    if not isinstance(obj, list):
        raise ValueError("timestamps_without_words parsed object is not a list")

    out: List[List[float]] = []
    for i, pair in enumerate(obj):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Bad timestamp pair at idx={i}: {pair!r}")
        a, b = pair
        out.append([float(a), float(b)])
    return out


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _move_file(src: Path, dst: Path, *, dry_run: bool, overwrite: bool) -> None:
    _ensure_parent_dir(dst)

    if dst.exists():
        if overwrite:
            if not dry_run:
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
        else:
            raise FileExistsError(f"Destination already exists (use --overwrite): {dst}")

    if dry_run:
        return

    shutil.move(str(src), str(dst))


# ============================================================
# [SRC RESOLUTION] robust source path resolver
#   Handles:
#     CSV `file` = "test/000001.mp4"
#     src_video_root = ".../Downloads/LAV-DF/test"
#   Naive join -> ".../test/test/000001.mp4" (wrong)
#
# Strategy:
#   1) Try src_root / rel
#   2) Try src_root / basename(rel)
#   3) If strip_prefix provided and rel starts with "<prefix>/", try src_root / stripped
# ============================================================
def resolve_src_video_path(src_video_root: Path, rel: str, *, strip_prefix: Optional[str] = None) -> Path:
    rel = (rel or "").strip().lstrip("/")
    if not rel:
        return src_video_root / rel

    p1 = src_video_root / rel
    if p1.exists():
        return p1

    base = os.path.basename(rel)
    p2 = src_video_root / base
    if p2.exists():
        return p2

    if strip_prefix:
        pref = strip_prefix.strip().strip("/")
        if pref and rel.startswith(pref + "/"):
            stripped = rel[len(pref) + 1 :]
            p3 = src_video_root / stripped
            if p3.exists():
                return p3

    return p1


# -----------------------------
# [CSV READ]
# -----------------------------
def read_csv_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


# -----------------------------
# [FILTER] ONLY file prefix
# -----------------------------
def filter_rows_by_file_prefix(rows: List[dict], required_prefix: str) -> List[dict]:
    pref = (required_prefix or "").strip()
    if not pref:
        return rows

    out: List[dict] = []
    for r in rows:
        rel = (r.get("file") or "").strip()
        if rel.startswith(pref):
            out.append(r)
    return out


# -----------------------------
# [SAMPLING]
# -----------------------------
def sample_rows_80_fake_20_real(
    rows: List[dict],
    *,
    fake_ratio: float,
    real_ratio: float,
    total_samples: Optional[int],
    seed: int,
) -> List[dict]:
    if abs((fake_ratio + real_ratio) - 1.0) > 1e-6:
        raise ValueError("fake_ratio + real_ratio must equal 1.0")

    fakes = [r for r in rows if _as_int_label(r.get("label", "")) == 1]
    reals = [r for r in rows if _as_int_label(r.get("label", "")) == 0]

    rng = random.Random(seed)
    rng.shuffle(fakes)
    rng.shuffle(reals)

    if total_samples is None:
        total_max = int(min(
            len(fakes) / max(fake_ratio, 1e-9),
            len(reals) / max(real_ratio, 1e-9),
        ))
        total_samples = total_max

    if total_samples <= 0:
        return []

    n_fake = int(round(total_samples * fake_ratio))
    n_real = total_samples - n_fake

    n_fake = min(n_fake, len(fakes))
    n_real = min(n_real, len(reals))

    # Fill remainder if clamped
    total_actual = n_fake + n_real
    if total_actual < total_samples:
        remaining = total_samples - total_actual
        extra_fake = min(remaining, len(fakes) - n_fake)
        n_fake += extra_fake
        remaining -= extra_fake
        if remaining > 0:
            extra_real = min(remaining, len(reals) - n_real)
            n_real += extra_real

    sampled = fakes[:n_fake] + reals[:n_real]
    rng.shuffle(sampled)
    return sampled


# -----------------------------
# [STAGE 1] MOVE FIRST; keep matched rows only
# -----------------------------
def stage1_move_and_keep_matched_rows(
    sampled_rows: List[dict],
    *,
    src_video_root: Path,
    dst_video_root: Path,
    strip_prefix_for_src: Optional[str],
    required_file_prefix: str,
    dry_run: bool,
    overwrite: bool,
    log_every: int,
) -> Tuple[List[dict], int, int, int, int]:
    """
    Returns:
      matched_rows, moved, missing, skipped, prefix_rejected

    prefix_rejected counts rows that do NOT start with required_file_prefix
    (should be 0 if you filtered correctly, but we enforce again for safety).
    """
    matched_rows: List[dict] = []
    moved = 0
    missing = 0
    skipped = 0
    prefix_rejected = 0

    pref = (required_file_prefix or "").strip()

    for i, r in enumerate(sampled_rows, start=1):
        rel = (r.get("file") or "").strip()
        if not rel:
            missing += 1
            continue

        # [ENFORCED] match ONLY "test/" (or whatever prefix you set)
        if pref and not rel.startswith(pref):
            prefix_rejected += 1
            continue

        src = resolve_src_video_path(src_video_root, rel, strip_prefix=strip_prefix_for_src).resolve()
        dst = (dst_video_root / rel).resolve()

        if not src.exists():
            missing += 1
            if i % log_every == 0:
                print(f"[stage1] i={i} moved={moved} missing={missing} skipped={skipped} (latest missing src: {src})")
            continue

        try:
            _move_file(src, dst, dry_run=dry_run, overwrite=overwrite)
            moved += 1
            matched_rows.append(r)
        except FileExistsError:
            skipped += 1
            # NOT included in matched_rows because you asked JSON only from moved/matched

        if i % log_every == 0:
            print(f"[stage1] i={i} moved={moved} missing={missing} skipped={skipped} (latest dst: {dst})")

    return matched_rows, moved, missing, skipped, prefix_rejected


# -----------------------------
# [STAGE 2] JSON only from matched/moved rows
# -----------------------------
def build_timestamps_json_from_matched_rows(matched_rows: List[dict]) -> Dict[str, List[List[float]]]:
    out: Dict[str, List[List[float]]] = {}

    for r in matched_rows:
        # Key must be video name with ext
        filename = (r.get("filename") or "").strip()
        if not filename:
            rel = (r.get("file") or "").strip()
            filename = os.path.basename(rel) if rel else ""

        if not filename:
            continue

        ts_str = r.get("timestamps_without_words", "")
        out[filename] = _safe_parse_timestamps_wo_words(ts_str)

    return out


def write_json(
    timestamps_map: Dict[str, List[List[float]]],
    *,
    json_out_dir: Path,
    json_filename: str,
    dry_run: bool,
    overwrite: bool,
) -> Path:
    json_out_dir = json_out_dir.resolve()
    json_out_dir.mkdir(parents=True, exist_ok=True)

    out_path = json_out_dir / json_filename

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"JSON already exists (use --overwrite): {out_path}")

    if dry_run:
        return out_path

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(timestamps_map, f, indent=2, ensure_ascii=False)

    return out_path


# -----------------------------
# [MAIN]
# -----------------------------
def run(cfg: SampleMoveConfig) -> None:
    print(f"[init] csv_path={cfg.csv_path}")
    print(f"[init] src_video_root={cfg.src_video_root}")
    print(f"[init] dst_video_root={cfg.dst_video_root}")
    print(f"[init] json_out_dir={cfg.json_out_dir}")
    print(f"[init] required_file_prefix={cfg.required_file_prefix!r}")
    print(f"[init] total_samples={cfg.total_samples} fake_ratio={cfg.fake_ratio} real_ratio={cfg.real_ratio} seed={cfg.seed}")
    print(f"[init] dry_run={cfg.dry_run} overwrite={cfg.overwrite} strip_prefix_for_src={cfg.strip_prefix_for_src}")

    rows_all = read_csv_rows(cfg.csv_path)
    print(f"[csv] rows_total={len(rows_all)}")

    # [CRITICAL] Filter ONLY test/ prefix files
    rows = filter_rows_by_file_prefix(rows_all, cfg.required_file_prefix)
    print(f"[filter] rows_with_prefix={len(rows)}")

    # Sample from filtered rows
    sampled = sample_rows_80_fake_20_real(
        rows,
        fake_ratio=cfg.fake_ratio,
        real_ratio=cfg.real_ratio,
        total_samples=cfg.total_samples,
        seed=cfg.seed,
    )
    s_fake = sum(1 for r in sampled if _as_int_label(r.get("label", "")) == 1)
    s_real = sum(1 for r in sampled if _as_int_label(r.get("label", "")) == 0)
    print(f"[sample] sampled_total={len(sampled)} sampled_fake={s_fake} sampled_real={s_real}")

    # Stage 1: move first; keep matched (moved) only
    matched_rows, moved, missing, skipped, prefix_rejected = stage1_move_and_keep_matched_rows(
        sampled,
        src_video_root=cfg.src_video_root,
        dst_video_root=cfg.dst_video_root,
        strip_prefix_for_src=cfg.strip_prefix_for_src,
        required_file_prefix=cfg.required_file_prefix,
        dry_run=cfg.dry_run,
        overwrite=cfg.overwrite,
        log_every=cfg.log_every,
    )
    m_fake = sum(1 for r in matched_rows if _as_int_label(r.get("label", "")) == 1)
    m_real = sum(1 for r in matched_rows if _as_int_label(r.get("label", "")) == 0)
    print(f"[stage1][done] moved={moved} missing={missing} skipped={skipped} matched_rows={len(matched_rows)} matched_fake={m_fake} matched_real={m_real} prefix_rejected={prefix_rejected}")

    if len(matched_rows) == 0:
        print("[stage2][skip] no moved/matched files -> not creating JSON")
        print("[done] complete")
        return

    # Stage 2: JSON only from moved/matched
    ts_map = build_timestamps_json_from_matched_rows(matched_rows)
    print(f"[stage2] timestamps_keys={len(ts_map)}")

    out_json = write_json(
        ts_map,
        json_out_dir=cfg.json_out_dir,
        json_filename=cfg.json_filename,
        dry_run=cfg.dry_run,
        overwrite=cfg.overwrite,
    )
    print(f"[stage2][done] json_path={out_json}")

    print("[done] complete")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='STRICT: only "test/" rows; sample 80/20; move matched files first; JSON from moved only.'
    )

    p.add_argument("--csv-path", type=str, required=True)
    p.add_argument("--src-video-root", type=str, required=True)
    p.add_argument("--dst-video-root", type=str, required=True)
    p.add_argument("--json-out-dir", type=str, required=True)

    p.add_argument("--total-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--json-filename", type=str, default="timestamps_without_words_sampled.json")
    p.add_argument("--log-every", type=int, default=500)

    # [ADDED] strict prefix config (default "test/")
    p.add_argument("--required-file-prefix", type=str, default="test/")

    # [ADDED] src resolution helper (default "test")
    p.add_argument("--strip-prefix-for-src", type=str, default="test")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    cfg = SampleMoveConfig(
        csv_path=Path(args.csv_path),
        src_video_root=Path(args.src_video_root),
        dst_video_root=Path(args.dst_video_root),
        json_out_dir=Path(args.json_out_dir),
        json_filename=args.json_filename,
        total_samples=args.total_samples,
        seed=args.seed,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        required_file_prefix=str(args.required_file_prefix),
        strip_prefix_for_src=(args.strip_prefix_for_src if args.strip_prefix_for_src else None),
        log_every=int(args.log_every),
    )

    run(cfg)
