#!/usr/bin/env python3
"""
Generate .gitignore rules for all subdirectories under thesis_main_files/data.

Usage:
    python generate_data_gitignore_rules.py

Then copy-paste the printed block into your .gitignore.
"""

import os
from pathlib import Path

# Adjust if your repo root moves
REPO_ROOT = Path("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean")
DATA_DIR = REPO_ROOT / "thesis_main_files" / "data"


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"DATA_DIR does not exist: {DATA_DIR}")

    print("# =====================================================================")
    print("# Auto-generated .gitignore rules for thesis_main_files/data")
    print("# Paste this block into your .gitignore at the repo root.")
    print("# It will:")
    print("#   - ignore all files under thesis_main_files/data")
    print("#   - but allow .gitkeep in each directory so the folder structure")
    print("#     can be kept in the repo if you add those files.")
    print("# =====================================================================")
    print()

    # Optional: a single top-level ignore for safety
    # (directories will be un-ignored below if needed)
    top_rel = DATA_DIR.relative_to(REPO_ROOT).as_posix()
    print(f"{top_rel}/**")
    print(f"!{top_rel}/")
    print()

    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        dir_path = Path(dirpath)
        rel = dir_path.relative_to(REPO_ROOT).as_posix()

        # Skip the DATA_DIR itself here since we already printed a top-level rule
        if dir_path == DATA_DIR:
            continue

        print(f"# Ignore files in {rel}, but allow the directory and its .gitkeep")
        # ignore all files directly in this directory
        print(f"{rel}/*")
        # allow a .gitkeep file if you create one
        print(f"!{rel}/.gitkeep")
        # allow the directory itself (so nested patterns still work)
        print(f"!{rel}/")
        print()


if __name__ == "__main__":
    main()

