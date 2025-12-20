
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple



# ----------------------------------------------------------------------
# Behavior toggles
# ----------------------------------------------------------------------

# Enforce end >= start
REQUIRE_NONDECREASING = True

# Deduplicate identical (start,end) pairs per clip
DEDUP_PAIRS = True

# If duplicate filename keys appear, keep the first occurrence
KEEP_FIRST_ON_DUPLICATE_KEY = True

# [EDIT] The CSV file (NOT a directory of many CSVs anymore)
CSV_PATH = Path(
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/raw/csv/lav_df/metadata/metadata.csv"
)

# [EDIT] Directory where the media files named in column "filename" live
# e.g. if filename is "000001.mp4", full path becomes:
#   FILES_DIR / "000001.mp4"
FILES_DIR = Path(
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/video_files/LAV_DF/video"
)

# [EDIT] Output JSON index file
OUT_JSON = Path(
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/DFDC/DFDC_timestamps_json_for_offline_training/"
    "DFDC_timestamps_json_for_offline_training.json"
)

# ======================================================================
# Helper functions
# ======================================================================

def _json_key_from_filename(filename: str) -> Optional[str]:
    """
    JSON key rule (STRICT):
      key = filename WITHOUT extension

    Example:
      "000001.mp4" -> "000001"
    """
    if not filename or not isinstance(filename, str):
        return None
    stem = Path(filename).stem.strip()
    return stem if stem else None


def _parse_timestamps_without_words(raw: str) -> Optional[List[Tuple[float, float]]]:
    """
    Parse the CSV column "timestamps_without_words".

    Expected string form:
      "[[0.0, 0.2], [0.2, 0.4], ...]"

    Returns:
      list of (start,end) tuples
      or None if malformed
    """
    if raw is None:
        return None

    raw = str(raw).strip()
    if raw == "":
        return None

    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(parsed, list):
        return None

    out: List[Tuple[float, float]] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            start = float(item[0])
            end = float(item[1])
        except (TypeError, ValueError):
            return None
        out.append((start, end))

    return out


def _normalize_pairs(
    pairs: List[Tuple[float, float]],
    *,
    require_nondecreasing: bool,
    dedup: bool,
) -> List[List[float]]:
    """
    Normalize timestamp pairs:
      - optionally drop end < start
      - optionally de-duplicate identical pairs
      - sort by start time
    """
    kept: List[List[float]] = []
    seen: set[Tuple[float, float]] = set()

    for start, end in pairs:
        if require_nondecreasing and end < start:
            continue

        if dedup:
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)

        kept.append([start, end])

    kept.sort(key=lambda t: t[0])
    return kept


# ======================================================================
# Core builder
# ======================================================================

def build_timestamp_index_from_csv(csv_path: Path) -> Dict[str, List[List[float]]]:
    """
    Build JSON mapping:
      { "<filename_stem>": [[start,end], ...], ... }

    Uses:
      - filename
      - timestamps_without_words
    """
    index: Dict[str, List[List[float]]] = {}

    total_rows = 0
    kept_rows = 0
    skipped_missing_filename = 0
    skipped_bad_timestamps = 0
    duplicate_keys = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # --------------------------------------------------------------
        # [CHECK] Required columns
        # --------------------------------------------------------------
        required_cols = {"filename", "timestamps_without_words"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV missing required columns: {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            total_rows += 1

            filename = row.get("filename", "")
            key = _json_key_from_filename(filename)
            if key is None:
                skipped_missing_filename += 1
                continue

            raw_pairs = row.get("timestamps_without_words")
            pairs = _parse_timestamps_without_words(raw_pairs)
            if pairs is None:
                skipped_bad_timestamps += 1
                continue

            normalized = _normalize_pairs(
                pairs,
                require_nondecreasing=REQUIRE_NONDECREASING,
                dedup=DEDUP_PAIRS,
            )

            if not normalized:
                skipped_bad_timestamps += 1
                continue

            # ----------------------------------------------------------
            # Duplicate key handling
            # ----------------------------------------------------------
            if key in index:
                duplicate_keys += 1
                if KEEP_FIRST_ON_DUPLICATE_KEY:
                    continue

            index[key] = normalized
            kept_rows += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=== Timestamp JSON build summary (timestamps_without_words) ===")
    print(f"CSV path                      : {csv_path}")
    print(f"Total rows                    : {total_rows}")
    print(f"JSON keys written             : {len(index)}")
    print(f"Rows kept                     : {kept_rows}")
    print(f"Skipped missing filename      : {skipped_missing_filename}")
    print(f"Skipped bad timestamps        : {skipped_bad_timestamps}")
    print(f"Duplicate keys encountered    : {duplicate_keys}")
    print(f"REQUIRE_NONDECREASING         : {REQUIRE_NONDECREASING}")
    print(f"DEDUP_PAIRS                   : {DEDUP_PAIRS}")
    print(f"KEEP_FIRST_ON_DUPLICATE_KEY   : {KEEP_FIRST_ON_DUPLICATE_KEY}")

    return index


# ======================================================================
# Main (hardcoded, no argparse)
# ======================================================================

def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    index = build_timestamp_index_from_csv(CSV_PATH)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nSaved timestamp index JSON to:\n  {OUT_JSON}")


if __name__ == "__main__":
    main()
