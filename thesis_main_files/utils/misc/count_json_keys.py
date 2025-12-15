#!/usr/bin/env python3
"""
count_timestamp_index_keys.py

Loads the JSON index produced by build_timestamp_index_hardcoded.py and
prints:

    - Number of audio keys (i.e., number of unique audio clips)
    - Optionally, total number of timestamp entries across all clips
"""

import json
from pathlib import Path


# ======================================================================
# [USER CONFIG] Hardcoded path — EDIT THIS
# ======================================================================

JSON_PATH = Path("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/AVSpeech/AVSpeech_timestamp_json_for_offline_training/AVSpeech_timestamp_json_for_offline_training.json")


# ======================================================================
# Main
# ======================================================================

def main():

    # --------------------------------------------------------------
    # Basic file check
    # --------------------------------------------------------------
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Index JSON not found: {JSON_PATH}")

    # --------------------------------------------------------------
    # Load the JSON into memory
    # --------------------------------------------------------------
    with JSON_PATH.open("r", encoding="utf-8") as f:
        index = json.load(f)

    # --------------------------------------------------------------
    # Count number of keys (unique audio files)
    # --------------------------------------------------------------
    num_audio_files = len(index)

    # --------------------------------------------------------------
    # Optional: count total timestamps across all audio files
    # (each timestamp is a [start, end] pair)
    # --------------------------------------------------------------
    total_timestamps = sum(len(v) for v in index.values())

    # --------------------------------------------------------------
    # Print summary
    # --------------------------------------------------------------
    print("=== Timestamp Index Summary ===")
    print(f"JSON file            : {JSON_PATH}")
    print(f"Unique audio files   : {num_audio_files}")
    print(f"Total timestamps     : {total_timestamps}")


if __name__ == "__main__":
    main()
