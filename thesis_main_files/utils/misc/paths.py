from pathlib import Path

# ---------------------------------------------------------
# 1) PROJECT ROOT RESOLUTION (always correct)
# src/paths.py  → parents[1] = project_root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------
# 2) DEFINE DIRECTORIES RELATIVE TO PROJECT ROOT
# ---------------------------------------------------------
DATA_DIR    = PROJECT_ROOT / "data" / "processed" / "video_files"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TRAIN_DIR   = PROJECT_ROOT / "train"
CORE_DIR    = PROJECT_ROOT / "core"
CORE_NPV_DIR = CORE_DIR / "NPVForensics"


def check_paths():
    paths = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATA_DIR": DATA_DIR,
        "SCRIPTS_DIR": SCRIPTS_DIR,
        "TRAIN_DIR": TRAIN_DIR,
        "CORE_DIR": CORE_DIR,
    }

    print("\n=== Path Sanity Check ===")
    for name, path in paths.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"{name:15} -> {path}   [{status}]")
    print("=========================\n")



### SANITY CHECK to check the existence of paths
# # Run when executing the file directly
# if __name__ == "__main__":
#     check_paths()

