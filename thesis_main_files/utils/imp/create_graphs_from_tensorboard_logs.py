#!/usr/bin/env python3
"""
plot_tb_csvs_by_stage_and_component.py

Enhancements over previous script:
1) Better filename parsing:
   - Extracts: stage (stage1/stage2/unknown), run_name, version, tag
   - Extracts "components" from run_name using a configurable keyword list
     (e.g., cpe, vacl, swin, without_swin, dfdc, lavdf, etc.)

2) Stage-aware output layout:
   OUT_ROOT/
     stage1/
       csvs/<component_bucket>/*.csv
       plots/<component_bucket>/*.png
       logs/plotter.log
     stage2/
       ...
     unknown/
       ...

3) Component-aware subdirectories:
   - Each file is routed to a bucket like:
       "vacl+cpe", "swin", "without_swin", "dfdc", "misc"
   - If multiple matched components -> joined with '+'
   - If none matched -> "misc"

Designed for Jupyter / notebook usage:
- Hardcode SRC_ROOT and OUT_ROOT below.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# [HARD-CODE PATHS HERE]
# =========================
SRC_ROOT = Path("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/results/tensorboard_csv")
OUT_ROOT = Path("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/results/saved_graphs_final")


# Copy CSVs into the organized structure
COPY_CSVS = True

# Overwrite existing copied CSVs and plots
OVERWRITE = True


# =========================
# Component keyword config
# =========================
# Add more keywords freely. Keep them lowercase.
# Rules:
# - Matching is done against run_name (lowercased), via token splitting and substring checks.
# - Some composite keywords (e.g. "without_swin") are useful to keep explicit.
COMPONENT_KEYWORDS: List[str] = [
    # Core modules / branches
    "cpe",
    "vacl",
    "ec",          # if you used "ec" in run names
    "info_nce",    # if you used these tokens
    "infonce",
    "projection",
    "align",

    # Backbone / variants
    "swin",
    "video_swin",
    "audio_swin",
    "without_swin",
    "no_swin",

    # Datasets / regimes
    "dfdc",
    "lavdf",
    "avspeech",
    "deepfaketimit",

    # Stage process markers
    "ssl",
    "finetune",
    "eval",
    "pretrain",
]

# If both appear, prefer the more specific bucket names by sorting matches with longer first.
# (Helps avoid "swin" swallowing "without_swin" conceptually in naming; we still keep both though.)
COMPONENT_KEYWORDS = sorted(set([k.lower() for k in COMPONENT_KEYWORDS]), key=len, reverse=True)


# =========================
# Logging
# =========================
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"tb_csv_plotter::{log_path.parent.name}")
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
@dataclass(frozen=True)
class ParsedName:
    stage: str              # "stage1" | "stage2" | "unknown"
    run_name: str           # everything after "run-" up to "_version_" or "-tag-"
    version: Optional[str]  # digits only if present
    tag: str                # scalar tag from TB export
    components: Tuple[str, ...]  # matched component keywords, canonicalized


# Handles both:
#   run-<RUN>_version_<N>-tag-<TAG>.csv
#   run-<RUN>-tag-<TAG>.csv
NAME_RE = re.compile(
    r"""
    ^run-(?P<run>.+?)
    (?:_version_(?P<version>\d+))?
    -tag-(?P<tag>.+?)
    \.csv$
    """,
    re.VERBOSE,
)


STAGE_RE = re.compile(r"(?:^|[_\-])(?P<stage>stage1|stage2)(?:[_\-]|$)", re.IGNORECASE)


def infer_stage(text: str) -> str:
    """
    Infer stage from filename OR run_name:
    - If contains "stage1" => stage1
    - If contains "stage2" => stage2
    - Else unknown
    """
    m = STAGE_RE.search(text)
    if not m:
        return "unknown"
    return m.group("stage").lower()


def tokenize_run(run_name: str) -> Set[str]:
    """
    Tokenize a run name using common separators.
    Example: stage2_finetune_main_without_swin_dfdc -> {stage2, finetune, main, without, swin, dfdc, ...}
    """
    parts = re.split(r"[^a-zA-Z0-9]+", run_name.lower())
    return {p for p in parts if p}


def extract_components(run_name: str, keywords: Iterable[str]) -> Tuple[str, ...]:
    """
    Extract component keywords from run_name.
    - Match if:
        a) keyword equals a token, OR
        b) keyword is a substring of run_name (useful for composite tokens)
    """
    rn = run_name.lower()
    tokens = tokenize_run(rn)

    hits: List[str] = []
    for kw in keywords:
        if kw in tokens or kw in rn:
            hits.append(kw)

    # De-dup while preserving order (keywords already sorted by length desc)
    seen = set()
    ordered = []
    for h in hits:
        if h not in seen:
            ordered.append(h)
            seen.add(h)

    return tuple(ordered)


def parse_tb_csv_name(filename: str) -> Optional[ParsedName]:
    m = NAME_RE.match(filename)
    if not m:
        return None

    run_name = m.group("run")
    version = m.group("version")
    tag = m.group("tag")

    stage = infer_stage(filename)
    if stage == "unknown":
        stage = infer_stage(run_name)

    comps = extract_components(run_name, COMPONENT_KEYWORDS)

    return ParsedName(
        stage=stage,
        run_name=run_name,
        version=version,
        tag=tag,
        components=comps,
    )


def component_bucket(components: Tuple[str, ...]) -> str:
    """
    Turn a component tuple into a stable folder name.
    - If empty => "misc"
    - Else join up to N keywords with '+'
    """
    if not components:
        return "misc"

    # Keep it readable; don’t create insane folder names
    MAX_COMPS = 5
    comps = components[:MAX_COMPS]
    return "+".join(comps)


# =========================
# CSV reading + plotting
# =========================
def load_scalar_csv(csv_path: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Expect columns like: Wall time, Step, Value (TensorBoard scalar export)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"Failed to read CSV: {csv_path} ({e})")
        return None

    cols = {c.strip().lower(): c for c in df.columns}
    if "step" not in cols or "value" not in cols:
        logger.warning(f"Skipping (missing Step/Value columns): {csv_path} columns={list(df.columns)}")
        return None

    out = df[[cols["step"], cols["value"]]].copy()
    out.columns = ["step", "value"]

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
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, logger: logging.Logger) -> bool:
    if dst.exists() and not OVERWRITE:
        logger.info(f"CSV exists, skipping copy (OVERWRITE=False): {dst}")
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    logger.info(f"Copied: {src} -> {dst}")
    return True


def main() -> None:
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT does not exist: {SRC_ROOT}")

    all_csvs = list(SRC_ROOT.rglob("*.csv"))

    # We'll keep per-stage log files.
    stage_roots = {
        "stage1": OUT_ROOT / "stage1",
        "stage2": OUT_ROOT / "stage2",
        "unknown": OUT_ROOT / "unknown",
    }
    for sr in stage_roots.values():
        ensure_dir(sr)

    loggers = {
        s: setup_logger(stage_roots[s] / "logs" / "plotter.log")
        for s in stage_roots
    }

    # Summary counters
    copied = 0
    plotted = 0
    skipped = 0
    unparsed = 0

    # Walk + process
    for src_csv in sorted(all_csvs):
        fname = src_csv.name

        parsed = parse_tb_csv_name(fname)
        if parsed is None:
            # Route unparsed into unknown/misc
            logger = loggers["unknown"]
            unparsed += 1
            logger.warning(f"Unrecognized filename pattern: {fname}  (routing to unknown/misc)")
            stage = "unknown"
            bucket = "misc"
            title = fname
            ylabel = "Value"
        else:
            stage = parsed.stage
            if stage not in stage_roots:
                stage = "unknown"

            logger = loggers[stage]
            bucket = component_bucket(parsed.components)

            v = f"version_{parsed.version}" if parsed.version is not None else "version_?"
            comp_str = bucket
            title = f"{parsed.run_name} | {v} | {parsed.tag} | comps:{comp_str}"
            ylabel = parsed.tag

        # Stage-aware dirs
        stage_root = stage_roots[stage]
        csv_out_dir = stage_root / "csvs" / bucket
        plot_out_dir = stage_root / "plots" / bucket

        # Copy (optional)
        if COPY_CSVS:
            dst_csv = csv_out_dir / fname
            try:
                did_copy = copy_file(src_csv, dst_csv, logger)
                if did_copy:
                    copied += 1
            except Exception as e:
                logger.warning(f"Copy failed: {src_csv} -> {dst_csv} ({e}). Will plot from source.")
                dst_csv = src_csv
        else:
            dst_csv = src_csv

        # Load + plot
        df = load_scalar_csv(dst_csv, logger)
        if df is None:
            skipped += 1
            continue

        out_png = plot_out_dir / (dst_csv.stem + ".png")
        plot_scalar_df(df, out_png, title=title, ylabel=ylabel, logger=logger)
        plotted += 1

    # Print global summary to stage2 logger (and others will have their own logs)
    # This is just a convenience line in each stage log.
    for stage, logger in loggers.items():
        logger.info("=== Global Summary ===")
        logger.info(f"Total CSVs scanned: {len(all_csvs)}")
        logger.info(f"Copied: {copied} | Plotted: {plotted} | Skipped: {skipped} | Unparsed: {unparsed}")
        logger.info(f"Outputs under: {stage_roots[stage]}")
        logger.info("Done.")


if __name__ == "__main__":
    main()
