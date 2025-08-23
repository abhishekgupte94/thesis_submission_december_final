#!/usr/bin/env python3
import os
import sys
import csv
import glob
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG
# =========================
video_root = "/Users/abhishekgupte_macbookpro/Downloads/LAV-DF"  # LAV-DF contains train/, test/, dev/
subsets_csv_dir = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/datasets/processed/csv_files/lav_df/new_setup/evaluate_files/evaluate/fake_files"  # where A_only_lt7p5.csv etc. live
subset_csvs = [
    "A_only_lt7p5.csv",
    "A_only_ge7p5.csv",
    "V_only_lt7p5.csv",
    "V_only_ge7p5.csv",
    "AV_both_lt7p5.csv",
    "AV_both_ge7p5.csv",
]
output_evaluate_files = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/datasets/processed/lav_df/new_setup/evaluate_files/evaluate/fake_files"  # destination root

MAX_WORKERS = 12           # tune for your disk/network
VERIFY_MD5 = False         # set True for extra safety (slower)
CLEAN_SOURCE_IF_DEST_MATCH = True  # if dest exists & verified equal, remove source to "finish" move

# =========================
# SETUP LOGGING
# =========================
os.makedirs(output_evaluate_files, exist_ok=True)
log_dir = os.path.join(output_evaluate_files, "logged_errors")
os.makedirs(log_dir, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
error_txt_path = os.path.join(log_dir, f"errors_{ts}.txt")
error_csv_path = os.path.join(log_dir, f"errors_{ts}.csv")
summary_csv_path = os.path.join(log_dir, f"summary_{ts}.csv")

error_txt = open(error_txt_path, "w", encoding="utf-8", buffering=1)

def log_error(subset, file_rel, src, dst, reason):
    msg = f"[{subset}] file={file_rel} | src={src} | dst={dst} | reason={reason}"
    error_txt.write(msg + "\n")
    errors_rows.append({
        "subset": subset,
        "file": file_rel,
        "source_path": src,
        "dest_path": dst,
        "reason": reason
    })

errors_rows = []
summary_rows = []

# =========================
# HELPERS
# =========================
def device_id(p: Path) -> int:
    try:
        return p.stat().st_dev
    except Exception:
        return -1

def same_device(src_path: Path, dst_path: Path) -> bool:
    # Compare device ids of parent folders (more stable)
    return device_id(src_path.parent) == device_id(dst_path.parent)

def md5sum(path: Path, chunk=1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return -1

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def build_src_path(row, subset_name):
    """
    Expect 'split' and 'file' columns. 'file' may include subdirs.
    Source = {video_root}/{split}/{file}
    """
    file_field = str(row["file"]).lstrip("./")
    split = str(row.get("split", "")).strip()
    if not split:
        # If no split provided, try to infer: search train/test/dev
        # Fallback: assume 'file' already contains split prefix
        parts = Path(file_field).parts
        if parts and parts[0] in ("train", "test", "dev"):
            return Path(video_root) / file_field
        # Last resort (not recommended): search
        for sp in ("train", "test", "dev"):
            candidate = Path(video_root) / sp / file_field
            if candidate.exists():
                return candidate
        # Not found; return canonical guess
        return Path(video_root) / file_field
    else:
        return Path(video_root) / split / file_field

def unique_dest_path(dst_dir: Path, base_name: str) -> Path:
    # Avoid collisions by suffixing __dupN if needed
    target = dst_dir / base_name
    if not target.exists():
        return target
    stem = target.stem
    ext = target.suffix
    k = 1
    while True:
        alt = dst_dir / f"{stem}__dup{k}{ext}"
        if not alt.exists():
            return alt
        k += 1

def safe_move_file(src: Path, dst: Path, subset_name: str, file_rel: str) -> str:
    """
    Robust move:
      - If same device: atomic os.replace (rename) after ensuring parent
      - If cross-device: copy -> verify -> atomic replace -> remove src
      - If dst exists: verify match; optionally remove src to finish
    Returns status: "moved" | "skipped" | "finished" | "error"
    """
    try:
        ensure_parent(dst)

        # If destination already exists, verify and optionally clean source
        if dst.exists():
            # Compare sizes (and md5 if enabled)
            dst_size = file_size(dst)
            src_size = file_size(src) if src.exists() else -1
            same = (src_size == dst_size and src_size >= 0)
            if same and VERIFY_MD5 and src.exists():
                same = (md5sum(src) == md5sum(dst))
            if same:
                if CLEAN_SOURCE_IF_DEST_MATCH and src.exists():
                    try:
                        src.unlink()  # remove source to "complete" move
                        return "finished"  # destination already had good copy
                    except Exception as e:
                        log_error(subset_name, file_rel, str(src), str(dst), f"rm_src_failed:{e!r}")
                        return "skipped"
                return "skipped"
            else:
                # Destination exists but differs; choose unique name
                dst = unique_dest_path(dst.parent, dst.name)

        # If source missing
        if not src.exists():
            log_error(subset_name, file_rel, str(src), str(dst), "source_missing")
            return "error"

        # Same-device: atomic rename
        if same_device(src, dst):
            os.replace(src, dst)  # atomic within same filesystem
            return "moved"

        # Cross-device: copy to temp then atomic commit
        tmp = dst.with_suffix(dst.suffix + ".partial")
        try:
            # Copy preserving metadata
            with src.open("rb") as fsrc, tmp.open("wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
            shutil.copystat(src, tmp, follow_symlinks=True)

            # Verify size (and md5 if enabled)
            ok = file_size(src) == file_size(tmp)
            if ok and VERIFY_MD5:
                ok = (md5sum(src) == md5sum(tmp))
            if not ok:
                tmp.unlink(missing_ok=True)
                log_error(subset_name, file_rel, str(src), str(dst), "verify_failed_after_copy")
                return "error"

            # Atomically replace destination, then remove source
            os.replace(tmp, dst)
            try:
                src.unlink()
            except Exception as e:
                # Not fatal; we already committed dst atomically
                log_error(subset_name, file_rel, str(src), str(dst), f"rm_src_failed:{e!r}")
            return "moved"
        except Exception as e:
            tmp.unlink(missing_ok=True)
            log_error(subset_name, file_rel, str(src), str(dst), f"copy_move_failed:{e!r}")
            return "error"

    except Exception as e:
        log_error(subset_name, file_rel, str(src), str(dst), f"unexpected:{e!r}")
        return "error"

def process_subset(csv_path: Path, subset_name: str):
    import pandas as pd
    moved = skipped = finished = errors = 0

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log_error(subset_name, None, str(csv_path), None, f"csv_read_error:{e!r}")
        summary_rows.append({"subset": subset_name, "moved": 0, "skipped": 0, "finished": 0, "errors": 1})
        return

    # Destination folder for this subset
    subset_out_dir = Path(output_evaluate_files) / subset_name
    subset_out_dir.mkdir(parents=True, exist_ok=True)

    # Build tasks
    tasks = []
    for _, row in df.iterrows():
        if "file" not in row or "split" not in row:
            log_error(subset_name, None, str(csv_path), str(subset_out_dir), "missing_required_columns(file/split)")
            continue
        file_rel = str(row["file"]).lstrip("./")
        src = build_src_path(row, subset_name)
        dst = unique_dest_path(subset_out_dir, Path(file_rel).name)  # preserve filename; handle collisions
        tasks.append((src, dst, subset_name, file_rel))

    # Parallel move
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(safe_move_file, src, dst, subset_name, file_rel): (src, dst, file_rel)
                for (src, dst, subset_name, file_rel) in tasks}
        for fut in as_completed(futs):
            try:
                status = fut.result()
            except Exception as e:
                src, dst, file_rel = futs[fut]
                log_error(subset_name, file_rel, str(src), str(dst), f"worker_exception:{e!r}")
                errors += 1
                continue
            if status == "moved":   moved += 1
            elif status == "skipped": skipped += 1
            elif status == "finished": finished += 1
            else: errors += 1

    summary_rows.append({
        "subset": subset_name,
        "moved": moved,
        "skipped": skipped,
        "finished": finished,
        "errors": errors
    })
    print(f"[{subset_name}] moved={moved} | skipped={skipped} | finished={finished} | errors={errors}")

# =========================
# MAIN
# =========================
def main():
    # Validate inputs
    if not Path(video_root).exists():
        print(f"❌ video_root does not exist: {video_root}")
        sys.exit(1)
    if not Path(subsets_csv_dir).exists():
        print(f"❌ subsets_csv_dir does not exist: {subsets_csv_dir}")
        sys.exit(1)

    # Process each subset CSV
    for csv_name in subset_csvs:
        csv_path = Path(subsets_csv_dir) / csv_name
        if not csv_path.exists():
            log_error(Path(csv_name).stem, None, str(csv_path), None, "csv_not_found")
            print(f"⚠️ CSV not found: {csv_path}")
            continue
        process_subset(csv_path, Path(csv_name).stem)

    # Write logs
    # Summary CSV
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "moved", "skipped", "finished", "errors"])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    # Errors CSV
    with open(error_csv_path, "w", newline="", encoding="utf-8") as f:
        if errors_rows:
            keys = ["subset", "file", "source_path", "dest_path", "reason"]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in errors_rows:
                w.writerow(row)

    error_txt.close()
    print("\n✅ DONE")
    print("📄 Summary:", summary_csv_path)
    print("🧾 Errors (txt):", error_txt_path)
    print("🧾 Errors (csv):", error_csv_path)

if __name__ == "__main__":
    main()
