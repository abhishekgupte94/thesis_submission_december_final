#!/usr/bin/env python3
"""
rewrite_timestamp_index_keys.py

Modify an existing JSON timestamp index in-place so that:

    "audio_train10009"  ->  "10009"

i.e. we strip the leading "audio_train" (and any following underscore)
from each key, while keeping the value (list of [start, end]) unchanged.

The script:
  1) Loads the JSON from JSON_PATH.
  2) Builds a new dict with updated keys.
  3) Optionally writes a backup.
  4) Overwrites the original JSON with the updated mapping.
"""

import json
from pathlib import Path
from typing import Dict, List


# ======================================================================
# [USER CONFIG] Hardcoded JSON path — EDIT THIS
# ======================================================================

JSON_PATH = Path("/path/to/avspeech_timestamp_index.json")

# If True, save a backup copy before overwriting.
SAVE_BACKUP = True
BACKUP_SUFFIX = ".bak"   # e.g. avspeech_timestamp_index.json.bak


# ======================================================================
# Helper: key normalisation
# ======================================================================

def shorten_key(key: str) -> str:
    """
    Strip the 'audio_train' prefix (and any optional underscore) from the key.

    Examples
    --------
    "audio_train10009"   -> "10009"
    "audio_train_10009"  -> "10009"

    If the key does not start with "audio_train", it is returned unchanged.
    """
    prefix = "audio_train"
    if not key.startswith(prefix):
        return key

    rest = key[len(prefix):]     # remove 'audio_train'
    rest = rest.lstrip("_")      # remove a leading underscore if present
    return rest or key  # fallback to original key if rest becomes empty


# ======================================================================
# Core logic
# ======================================================================

def rewrite_keys_in_json(json_path: Path) -> None:
    """
    Load the JSON at `json_path`, rewrite keys, and overwrite the file.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # --------------------------------------------------------------
    # Load existing index
    # --------------------------------------------------------------
    with json_path.open("r", encoding="utf-8") as f:
        index: Dict[str, List[List[float]]] = json.load(f)

    print(f"[INFO] Loaded JSON with {len(index)} keys from: {json_path}")

    # --------------------------------------------------------------
    # Build a new mapping with shortened keys
    # --------------------------------------------------------------
    new_index: Dict[str, List[List[float]]] = {}

    for old_key, value in index.items():
        new_key = shorten_key(old_key)

        if new_key in new_index:
            # If this happens, you likely have collisions like:
            # "audio_train10009" and "audio_train_10009"
            # both mapping to "10009".
            # Adjust logic here if you want to merge or handle differently.
            raise ValueError(
                f"Key collision after renaming: {old_key!r} -> {new_key!r} "
                f"(already used). You may need to adjust shorten_key()."
            )

        new_index[new_key] = value

    print(f"[INFO] After rewrite, JSON will have {len(new_index)} keys.")

    # --------------------------------------------------------------
    # Optional: save backup before overwriting
    # --------------------------------------------------------------
    if SAVE_BACKUP:
        backup_path = json_path.with_suffix(json_path.suffix + BACKUP_SUFFIX)
        backup_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(f"[INFO] Backup saved to: {backup_path}")

    # --------------------------------------------------------------
    # Overwrite original file with new mapping
    # --------------------------------------------------------------
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)

    print(f"[DONE] Rewritten JSON saved to: {json_path}")


# ======================================================================
# Entry point
# ======================================================================
JSON_PATH = Path("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/AVSpeech/AVSpeech_timestamp_json_for_offline_training/AVSpeech_timestamp_json_for_offline_training.json")
def main():
    rewrite_keys_in_json(JSON_PATH)


if __name__ == "__main__":
    main()
