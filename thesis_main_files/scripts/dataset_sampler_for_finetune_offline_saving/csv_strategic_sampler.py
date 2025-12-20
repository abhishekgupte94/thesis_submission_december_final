from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union
import os
import shutil

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class StrategicSampleSpec:
    """
    Strategic sampling configuration.
    """
    n_total: int
    seed: int = 42

    # Ratios
    label_ratio_fake: float = 0.80      # 80% FAKE (label=1)
    short_ratio: float = 0.80           # 80% duration <= short_max_seconds
    short_max_seconds: float = 7.5

    # Behavior
    allow_backfill: bool = True
    move_files: bool = True
    overwrite_existing_in_dst: bool = False
    dry_run: bool = False


# =============================================================================
# Internal helpers
# =============================================================================

def _scan_directory(root: Path) -> Dict[str, Path]:
    """
    Recursively scan directory and map filename -> full path.
    """
    mapping: Dict[str, Path] = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            mapping[fn] = Path(dirpath) / fn
    return mapping


def _choose(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0]
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed, replace=False)


# =============================================================================
# Main API
# =============================================================================

def strategic_sample_and_move(
    *,
    csv_in_path: Union[str, Path],
    csv_out_path: Union[str, Path],
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    duration_col: str,
    spec: StrategicSampleSpec,
    filename_col: str = "filename",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Sample rows from CSV using:
      - 80% FAKE / 20% REAL
      - 80% duration <= 7.5s
    Then move ONLY those sampled videos (max n_total).
    """

    csv_in_path = Path(csv_in_path)
    csv_out_path = Path(csv_out_path)
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    df = pd.read_csv(csv_in_path)

    # Validate columns
    for col in (filename_col, label_col, duration_col):
        if col not in df.columns:
            raise KeyError(f"CSV missing column '{col}'")

    # Normalize filename to basename
    df = df.copy()
    df[filename_col] = df[filename_col].astype(str).apply(lambda x: Path(x).name)

    # Scan filesystem once
    file_map = _scan_directory(src_dir)

    # Keep only files that exist
    df["__src_path"] = df[filename_col].map(file_map)
    df = df[df["__src_path"].notna()].copy()

    # Clean duration
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df = df[df[duration_col].notna()].copy()

    # Buckets
    df["__is_fake"] = df[label_col].astype(int) == 1
    df["__is_short"] = df[duration_col] <= spec.short_max_seconds

    # Targets
    n_total = spec.n_total
    n_fake = int(round(n_total * spec.label_ratio_fake))
    n_real = n_total - n_fake

    n_fake_short = int(round(n_fake * spec.short_ratio))
    n_fake_other = n_fake - n_fake_short
    n_real_short = int(round(n_real * spec.short_ratio))
    n_real_other = n_real - n_real_short

    # Buckets
    fs = df[df["__is_fake"] & df["__is_short"]]
    fo = df[df["__is_fake"] & ~df["__is_short"]]
    rs = df[~df["__is_fake"] & df["__is_short"]]
    ro = df[~df["__is_fake"] & ~df["__is_short"]]

    seed = spec.seed

    sampled = pd.concat(
        [
            _choose(fs, n_fake_short, seed + 1),
            _choose(fo, n_fake_other, seed + 2),
            _choose(rs, n_real_short, seed + 3),
            _choose(ro, n_real_other, seed + 4),
        ],
        ignore_index=True,
    )

    # Backfill if needed
    if spec.allow_backfill and len(sampled) < n_total:
        remaining = n_total - len(sampled)
        pool = df.drop(sampled.index, errors="ignore")
        sampled = pd.concat(
            [sampled, _choose(pool, remaining, seed + 10)],
            ignore_index=True,
        )

    # Enforce unique filenames and hard cap
    sampled = sampled.drop_duplicates(subset=[filename_col])
    if len(sampled) > n_total:
        sampled = sampled.sample(n=n_total, random_state=seed)

    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Write output CSV (original schema only)
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.drop(
        columns=["__src_path", "__is_fake", "__is_short"],
        errors="ignore",
    ).to_csv(csv_out_path, index=False)

    # Move files
    dst_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0

    if not spec.dry_run:
        for _, row in sampled.iterrows():
            src = Path(row["__src_path"])
            dst = dst_dir / src.name

            if dst.exists() and not spec.overwrite_existing_in_dst:
                skipped += 1
                continue

            if dst.exists():
                dst.unlink()

            if spec.move_files:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)

            moved += 1

    stats = {
        "rows_found_on_disk": len(df),
        "rows_sampled": len(sampled),
        "moved": moved,
        "skipped_existing": skipped,
        "fake_count": int((sampled[label_col] == 1).sum()),
        "real_count": int((sampled[label_col] == 0).sum()),
    }

    return sampled, stats
