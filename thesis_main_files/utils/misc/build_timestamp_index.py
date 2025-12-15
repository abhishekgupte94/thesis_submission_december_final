#!/usr/bin/env python3
"""
build_timestamp_index_hardcoded.py

Scan a directory of CSV files with columns:

    audio_file,segment_index,word,start,end

For each audio_file, collect a list of [start, end] timestamps (floats),
skipping:
  - rows with non-numeric start/end
  - duplicate (start, end, word) triplets for that audio_file

The final JSON mapping has keys like "audio_<id>", where <id> is taken
from the original filename after stripping a leading "trim_" and
removing the extension, e.g.:

    "trim_audio_train123.wav" --> "audio_train123"

This version uses **hardcoded** paths for:
  - CSV_DIR      : where the *_words.csv files live
  - OUT_JSON     : where the final index JSON is written
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


# ======================================================================
# [USER CONFIG] Hardcoded paths — EDIT THESE
# ======================================================================

# Directory containing your per-clip CSV files (e.g. *_words.csv)
CSV_DIR = Path("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/AVSpeech/AVSpeech_timestamps_csv")


# Output JSON index file
OUT_JSON = Path("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/AVSpeech/AVSpeech_timestamp_json_for_offline_training/AVSpeech_timestamp_json_for_offline_training.json")


# ======================================================================
# Helper functions
# ======================================================================

def normalise_audio_key(audio_filename: str) -> str:
    """
    Convert an audio filename like:
        'trim_audio_train123.wav'
    into a key like:
        'audio_train123'

    Rules:
      1) Strip a leading 'trim_' if present.
      2) Strip the file extension.

    You can adjust this if your naming pattern is different.
    """
    # 1) Strip leading 'trim_' if present
    if audio_filename.startswith("trim_"):
        audio_filename = audio_filename[len("trim_"):]  # remove 'trim_'

    # 2) Remove extension (everything after last '.')
    stem = audio_filename.rsplit(".", maxsplit=1)[0]

    # We now expect something like 'audio_train123'
    return stem


def is_float(value: str) -> bool:
    """
    Return True if `value` can be parsed as float, otherwise False.

    Used to guard against 'weird' rows where start/end are non-numeric.
    """
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


# ======================================================================
# Core builder
# ======================================================================

def build_timestamp_index(csv_dir: Path) -> Dict[str, List[List[float]]]:
    """
    Walk `csv_dir` for all *.csv files, parse them, and build:

        {
          "audio_train123": [[start, end], [start, end], ...],
          ...
        }

    while skipping:
      - rows with non-numeric start/end
      - duplicate (start, end, word) per audio_key
    """
    timestamp_index: Dict[str, List[List[float]]] = {}
    # For deduplicating rows per audio clip
    seen_triplets: Dict[str, set[Tuple[str, str, str]]] = {}

    total_files = 0
    total_rows = 0
    skipped_non_numeric = 0
    skipped_duplicates = 0

    # ------------------------------------------------------------------
    # Iterate over all CSV files in the directory
    # ------------------------------------------------------------------
    for csv_path in sorted(csv_dir.glob("*.csv")):
        total_files += 1

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # ----------------------------------------------------------
            # Each row represents one word occurrence and its timestamps
            # ----------------------------------------------------------
            for row in reader:
                total_rows += 1

                audio_file = row.get("audio_file")
                start_str = row.get("start")
                end_str = row.get("end")
                word = row.get("word", "")

                # ------------------------------------------------------
                # Basic sanity checks: missing key fields → skip
                # ------------------------------------------------------
                if not audio_file or start_str is None or end_str is None:
                    skipped_non_numeric += 1
                    continue

                # ------------------------------------------------------
                # Skip rows with non-numeric start/end
                # (handles 'weird' rows you mentioned)
                # ------------------------------------------------------
                if not (is_float(start_str) and is_float(end_str)):
                    skipped_non_numeric += 1
                    continue

                # ------------------------------------------------------
                # Normalise the key for this audio file
                # e.g. 'trim_audio_train123.wav' → 'audio_train123'
                # ------------------------------------------------------
                audio_key = normalise_audio_key(audio_file)

                # Prepare containers for this audio_key if new
                if audio_key not in timestamp_index:
                    timestamp_index[audio_key] = []
                    seen_triplets[audio_key] = set()

                # ------------------------------------------------------
                # Deduplicate by (start_str, end_str, word)
                # If there are unnecessary duplicates, we skip them.
                # ------------------------------------------------------
                triplet = (start_str, end_str, word)
                if triplet in seen_triplets[audio_key]:
                    skipped_duplicates += 1
                    continue
                seen_triplets[audio_key].add(triplet)

                # Parse to float
                start = float(start_str)
                end = float(end_str)

                # Append [start, end] pair for this audio_key
                timestamp_index[audio_key].append([start, end])

    # ------------------------------------------------------------------
    # Optional: sort each list by start time for sanity
    # ------------------------------------------------------------------
    for key in timestamp_index:
        timestamp_index[key].sort(key=lambda t: t[0])

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------
    print("=== Timestamp index build summary ===")
    print(f"CSV directory    : {csv_dir}")
    print(f"CSV files found  : {total_files}")
    print(f"Total rows       : {total_rows}")
    print(f"Valid entries    : {sum(len(v) for v in timestamp_index.values())}")
    print(f"Skipped non-numeric/missing start/end : {skipped_non_numeric}")
    print(f"Skipped duplicates                    : {skipped_duplicates}")
    print("Number of audio keys in index        :",
          len(timestamp_index))

    return timestamp_index


# ======================================================================
# Main entry point (no argparse, uses hardcoded config above)
# ======================================================================

def main():
    if not CSV_DIR.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {CSV_DIR}")

    index = build_timestamp_index(CSV_DIR)

    # Ensure parent directory exists
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nSaved timestamp index to: {OUT_JSON}")


if __name__ == "__main__":
    main()
