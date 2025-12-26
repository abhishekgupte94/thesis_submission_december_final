#!/usr/bin/env python3
"""
plot_tb_csvs_stage2.py

Walk a directory of TensorBoard-downloaded scalar CSVs, collect Stage-2 related CSVs
into a common folder called "stage2", then generate one plot per CSV:

- X axis: Step
- Y axis: Value
- Title/labels inferred from filename tokens (run / version / tag)
- Output: stage2/plots/<same-name>.png

Designed for Jupyter usage: hardcode the paths below and run the cell/script.

Filename example:
  run-stage2_finetune_main_without_swin_dfdc_version_1-tag-val_loss_cls_step.csv
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# [HARD-CODE PATHS HERE]
# =========================
# Directory that contains all your downloaded TB CSVs (possibly nested)
SRC_ROOT = Path("")

# Where to put the unified folder "stage2" (created if missing)
OUT_ROOT = Path("")

# Only pick CSVs whose filename contains this substring (edit if you want more/less strict)
STAGE2_NAME_SUBSTR = "stage2"

# If True: also copy the CSVs into OUT_ROOT/stage2/csvs/
COPY_CSCSV_INTO_STAGE2 = True

# If True: overwrite existing copied CSVs and plots
OVERWRITE = True


# =========================
# Logging
# =========================
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("tb_csv_plotter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# =========================
# Filename parsing
# =========================
@dataclass
class ParsedName:
    run: str
    version: Optional[str]
    tag: str


# Matches: run-<RUN>_version_<N>-tag-<TAG>.csv
# Also tolerates different separators slightly.
NAME_RE = re.compile(
    r"""
    ^run-(?P<run>.+?)               # run name (non-greedy)
    (?:_version_(?P<version>\d+))?  # optional version
    -tag-(?P<tag>.+?)               # tag
    \.csv$                          # extension
    """,
    re.VERBOSE,
)


def parse_tb_csv_name(filename: str) -> Optional[ParsedName]:
    m = NAME_RE.match(filename)
    if not m:
        return None
    run = m.group("run")
    version = m.group("version")
    tag = m.group("tag")
    return ParsedName(run=run, version=version, tag=tag)


# =========================
# CSV reading + plotting
# =========================
def load_scalar_csv(csv_path: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Expect columns like: Wall time, Step, Value
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"Failed to read CSV: {csv_path} ({e})")
        return None

    # Normalize column names
    cols = {c.strip().lower(): c for c in df.columns}
    required = ["step", "value"]
    if not all(k in cols for k in required):
        logger.warning(f"Skipping (missing Step/Value columns): {csv_path} columns={list(df.columns)}")
        return None

    # Make a clean canonical DF
    step_col = cols["step"]
    value_col = cols["value"]
    out = df[[step_col, value_col]].copy()
    out.columns = ["step", "value"]

    # Coerce
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["step", "value"]).sort_values("step")

    if len(out) == 0:
        logger.warning(f"Skipping (no valid rows after cleaning): {csv_path}")
        return None

    return out


def plot_scalar_df(
    df: pd.DataFrame,
    out_png: Path,
    title: str,
    xlabel: str = "Step",
    ylabel: str = "Value",
    logger: Optional[logging.Logger] = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["step"].to_numpy(), df["value"].to_numpy())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if out_png.exists() and not OVERWRITE:
        if logger:
            logger.info(f"Plot exists, skipping (OVERWRITE=False): {out_png}")
        plt.close()
        return

    plt.savefig(out_png, dpi=200)
    plt.close()

    if logger:
        logger.info(f"Saved plot: {out_png}")


# =========================
# Main routine
# =========================
def main() -> None:
    stage2_dir = OUT_ROOT / "stage2"
    csv_out_dir = stage2_dir / "csvs"
    plot_out_dir = stage2_dir / "plots"
    log_path = stage2_dir / "plot_tb_csvs_stage2.log"

    logger = setup_logger(log_path)
    logger.info("=== TensorBoard CSV Plotter (Stage2) ===")
    logger.info(f"SRC_ROOT: {SRC_ROOT}")
    logger.info(f"OUT_ROOT: {OUT_ROOT}")
    logger.info(f"Filter substring: {STAGE2_NAME_SUBSTR!r}")
    logger.info(f"COPY_CSV_INTO_STAGE2: {COPY_CSCSV_INTO_STAGE2}")
    logger.info(f"OVERWRITE: {OVERWRITE}")

    if not SRC_ROOT.exists():
        logger.error(f"SRC_ROOT does not exist: {SRC_ROOT}")
        return

    stage2_dir.mkdir(parents=True, exist_ok=True)
    if COPY_CSCSV_INTO_STAGE2:
        csv_out_dir.mkdir(parents=True, exist_ok=True)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    # Find candidate CSVs
    all_csvs = list(SRC_ROOT.rglob("*.csv"))
    logger.info(f"Found {len(all_csvs)} total CSV files under SRC_ROOT")

    # Filter to stage2 by filename substring
    cand = [p for p in all_csvs if STAGE2_NAME_SUBSTR in p.name]
    logger.info(f"Filtered to {len(cand)} CSV files containing {STAGE2_NAME_SUBSTR!r} in filename")

    copied_count = 0
    plotted_count = 0
    skipped_count = 0

    for src_csv in sorted(cand):
        fname = src_csv.name

        parsed = parse_tb_csv_name(fname)
        if parsed is None:
            logger.warning(f"Unrecognized filename pattern, will still try plotting: {fname}")

        # Decide destination CSV path (optional)
        if COPY_CSCSV_INTO_STAGE2:
            dst_csv = csv_out_dir / fname
            if dst_csv.exists() and not OVERWRITE:
                logger.info(f"CSV exists, skipping copy (OVERWRITE=False): {dst_csv}")
            else:
                try:
                    shutil.copy2(src_csv, dst_csv)
                    copied_count += 1
                    logger.info(f"Copied: {src_csv} -> {dst_csv}")
                except Exception as e:
                    logger.warning(f"Failed to copy {src_csv} -> {dst_csv} ({e})")
                    # even if copy fails, try plotting from source
                    dst_csv = src_csv
        else:
            dst_csv = src_csv

        df = load_scalar_csv(dst_csv, logger)
        if df is None:
            skipped_count += 1
            continue

        # Build plot title from parsed tokens if possible
        if parsed is not None:
            v = f"version_{parsed.version}" if parsed.version is not None else "version_?"
            title = f"{parsed.run} | {v} | {parsed.tag}"
            # Use tag as ylabel if it looks like a scalar name
            ylabel = parsed.tag
        else:
            title = fname
            ylabel = "Value"

        out_png = plot_out_dir / (dst_csv.stem + ".png")
        plot_scalar_df(df, out_png, title=title, ylabel=ylabel, logger=logger)
        plotted_count += 1

    logger.info("=== Summary ===")
    logger.info(f"Copied CSVs: {copied_count}")
    logger.info(f"Plotted CSVs: {plotted_count}")
    logger.info(f"Skipped CSVs: {skipped_count}")
    logger.info(f"Stage2 outputs: {stage2_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
