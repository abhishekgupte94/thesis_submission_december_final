# scripts/utils/convert_train_videos_to_train_audio.py
# ============================================================
# [NEW SCRIPT] Convert videos already moved into the TRAIN split
# into TRAIN audio (.wav) using ffmpeg.
#
# This replaces the previous "audio mover" CSV-based copy approach.
# It does NOT use the metadata CSV for selection — it scans the TRAIN video dir.
#
# Key properties (copied/reused from your pasted guide, only relevant bits):
#   - ffmpeg-based conversion to PCM 16-bit WAV, 16kHz, mono
#   - parallel conversion via ThreadPool
#   - resume-safe logging:
#       - success CSV: <LOG_DIR>/audio_success.csv
#       - failed  CSV: <LOG_DIR>/audio_failed.csv
#     Any video already present in either log is skipped.
#   - cap: processes at most 15,000 "new" videos (after skipping logged ones)
#
# NOTES
# -----
# - Input videos live in:  .../data/processed/splits/train
# - Output wavs go to:     .../data/processed/splits/train/audio
# - If your TRAIN video files are nested in subfolders, set RECURSIVE=True.

from __future__ import annotations

import csv
import os
import subprocess
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# =============================================================================
# User paths (FINAL)
# =============================================================================

TRAIN_VIDEO_DIR = (
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/data/processed/splits/train/video"
)

TRAIN_AUDIO_DIR = (
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/data/processed/splits/train/audio"
)

LOG_DIR = (
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/temp_files/logs_for_audio_conversion_train_split"
)


# =============================================================================
# Helpers (reused)
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_ffmpeg(cmd: List[str]) -> bool:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        # ffmpeg not installed / not on PATH
        return False


def _append_csv_row(csv_path: str, header: List[str], row: List[str]) -> None:
    ensure_header = not os.path.exists(csv_path)
    ensure_dir(str(Path(csv_path).parent))
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if ensure_header:
            w.writerow(header)
        w.writerow(row)


def _load_logged_video_paths(csv_path: str, video_path_col: str = "video_path") -> set:
    """
    Load a set of video_path values from a CSV log.
    If the file doesn't exist or is malformed, returns an empty set.
    """
    if not os.path.exists(csv_path):
        return set()

    logged = set()
    try:
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames or video_path_col not in r.fieldnames:
                return set()
            for row in r:
                vp = (row.get(video_path_col) or "").strip()
                if vp:
                    logged.add(vp)
    except Exception:
        return set()

    return logged


# =============================================================================
# Single-file conversion (reused)
# =============================================================================

def convert_video_to_audio(
    video_path: str,
    audio_path: str,
    sr: int = 16000,
    mono: bool = True,
    overwrite: bool = True,
    sanity_check_wav: bool = True,
) -> Tuple[bool, str]:
    """
    Returns:
        (ok, error_message)
    """
    ensure_dir(os.path.dirname(audio_path) or ".")

    if not os.path.exists(video_path):
        return False, "missing video"

    ffmpeg_cmd = ["ffmpeg"]
    if overwrite:
        ffmpeg_cmd.append("-y")

    ffmpeg_cmd += [
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
    ]

    if mono:
        ffmpeg_cmd += ["-ac", "1"]

    ffmpeg_cmd.append(audio_path)

    ok = run_ffmpeg(ffmpeg_cmd)
    if not ok:
        return False, "ffmpeg failed (is ffmpeg installed and on PATH?)"

    if sanity_check_wav:
        try:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 44:
                return False, "empty/invalid wav"
        except OSError:
            return False, "wav filesize check failed"

    return True, ""


# =============================================================================
# Directory → audio (parallel, capped, logged, resume-safe)
# =============================================================================

def convert_video_dir_to_audio_parallel(
    video_dir: str,
    audio_dir: str,
    log_dir: str,
    *,
    exts: Sequence[str] = (".mp4", ".mkv", ".mov", ".webm", ".m4v"),
    num_workers: int = 8,
    sr: int = 16000,
    mono: bool = True,
    overwrite: bool = True,
    sanity_check_wav: bool = True,
    recursive: bool = False,
    max_videos: Optional[int] = 15000,  # [DEFAULT] move only 15,000 new conversions
    success_csv_name: str = "audio_success.csv",
    failed_csv_name: str = "audio_failed.csv",
) -> Dict[str, object]:
    """
    Converts videos in `video_dir` to WAV in `audio_dir` in parallel.

    Resume-safe logging:
      - Writes successes to: <log_dir>/<success_csv_name>
      - Writes failures  to: <log_dir>/<failed_csv_name>
      - SKIPS any video already present in either CSV (by absolute path).

    Output naming:
      - <audio_dir>/<video_stem>.wav
    """
    vdir = Path(video_dir)
    adir = Path(audio_dir)
    ldir = Path(log_dir)

    if not vdir.is_dir():
        raise ValueError(f"Invalid video_dir: {video_dir}")

    ensure_dir(str(adir))
    ensure_dir(str(ldir))

    success_csv = str(ldir / success_csv_name)
    failed_csv = str(ldir / failed_csv_name)

    done_success = _load_logged_video_paths(success_csv, video_path_col="video_path")
    done_failed = _load_logged_video_paths(failed_csv, video_path_col="video_path")
    done_all = done_success.union(done_failed)

    # Deterministic enumeration
    globber = vdir.rglob("*") if recursive else vdir.glob("*")
    videos_all = sorted(
        [p for p in globber if p.is_file() and p.suffix.lower() in set(exts)],
        key=lambda p: p.name,
    )

    def _norm(p: Path) -> str:
        return str(p.resolve())

    # Filter already-logged
    videos_new = [p for p in videos_all if _norm(p) not in done_all]

    # Cap AFTER filtering => caps "new work"
    if max_videos is not None:
        videos_new = videos_new[:max_videos]

    if not videos_new:
        return {
            "found_total": len(videos_all),
            "already_logged": len(done_all),
            "scheduled": 0,
            "ok_count": 0,
            "fail_count": 0,
            "success_csv": success_csv,
            "failed_csv": failed_csv,
            "ok": [],
            "failed": [],
        }

    def _worker(v: Path) -> Tuple[bool, str, str, str]:
        video_path = _norm(v)
        audio_path = str((adir / f"{v.stem}.wav").resolve())
        ok, err = convert_video_to_audio(
            video_path=video_path,
            audio_path=audio_path,
            sr=sr,
            mono=mono,
            overwrite=overwrite,
            sanity_check_wav=sanity_check_wav,
        )
        return ok, video_path, audio_path, err

    pool = ThreadPool(num_workers)
    results = pool.imap_unordered(_worker, videos_new)

    ok_rows: List[Tuple[str, str]] = []
    fail_rows: List[Tuple[str, str]] = []

    for ok, video_path, audio_path, err in results:
        if ok:
            ok_rows.append((video_path, audio_path))
            print(f"[OK] {Path(video_path).name} → {Path(audio_path).name}")
        else:
            fail_rows.append((video_path, err))
            print(f"[FAIL] {Path(video_path).name} ({err})")

    pool.close()
    pool.join()

    # Write logs once at end (avoids thread locking issues)
    for video_path, audio_path in ok_rows:
        _append_csv_row(
            success_csv,
            header=["video_path", "audio_path"],
            row=[video_path, audio_path],
        )

    for video_path, err in fail_rows:
        _append_csv_row(
            failed_csv,
            header=["video_path", "error"],
            row=[video_path, err],
        )

    return {
        "found_total": len(videos_all),
        "already_logged": len(done_all),
        "scheduled": len(videos_new),
        "ok_count": len(ok_rows),
        "fail_count": len(fail_rows),
        "success_csv": success_csv,
        "failed_csv": failed_csv,
        "ok": ok_rows,
        "failed": fail_rows,
    }


# =============================================================================
# Usage (no argparse)
# =============================================================================

if __name__ == "__main__":
    summary = convert_video_dir_to_audio_parallel(
        video_dir=TRAIN_VIDEO_DIR,
        audio_dir=TRAIN_AUDIO_DIR,
        log_dir=LOG_DIR,
        num_workers=6,
        # max_videos=15000,    # cap NEW videos processed this run
        recursive=False,     # set True if train videos are in nested dirs
        overwrite=True,      # ffmpeg -y (replace wav if exists)
    )

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if k in ("ok", "failed"):
            continue
        print(f"{k}: {v}")
