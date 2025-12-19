from __future__ import annotations

import os
import subprocess
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# -----------------------------
# Helpers
# -----------------------------
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
        if result.returncode != 0:
            print("[ffmpeg] ERROR:", " ".join(cmd))
            print(result.stderr.decode("utf-8", errors="ignore"))
            return False
        return True
    except FileNotFoundError:
        print("[ffmpeg] ERROR: ffmpeg not found in PATH.")
        return False


# -----------------------------
# Single file conversion
# -----------------------------
def convert_video_to_audio(
    video_path: str,
    audio_path: str,
    sr: int = 16000,
    mono: bool = True,
    overwrite: bool = True,
    sanity_check_wav: bool = True,
) -> bool:
    ensure_dir(os.path.dirname(audio_path) or ".")

    if not os.path.exists(video_path):
        print(f"[ERROR] Missing video: {video_path}")
        return False

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

    if not run_ffmpeg(ffmpeg_cmd):
        return False

    if sanity_check_wav:
        try:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 44:
                print(f"[ERROR] Invalid WAV: {audio_path}")
                return False
        except OSError:
            return False

    return True


# -----------------------------
# Directory → audio (parallel)
# -----------------------------
def convert_video_dir_to_audio_parallel(
    video_dir: str,
    audio_dir: str,
    *,
    exts: Sequence[str] = (".mp4", ".mkv", ".mov", ".webm", ".m4v"),
    num_workers: int = 8,
    sr: int = 16000,
    mono: bool = True,
    overwrite: bool = True,
    sanity_check_wav: bool = True,
    recursive: bool = False,
) -> Dict[str, object]:
    """
    Convert all videos in `video_dir` to WAVs in `audio_dir`
    using the SAME filename (stem preserved).

    Example:
        video_001.mp4 → video_001.wav
    """
    vdir = Path(video_dir)
    adir = Path(audio_dir)
    ensure_dir(str(adir))

    if not vdir.is_dir():
        raise ValueError(f"Invalid video_dir: {video_dir}")

    globber = vdir.rglob("*") if recursive else vdir.glob("*")
    videos = [
        p for p in globber
        if p.is_file() and p.suffix.lower() in exts
    ]

    def _worker(v: Path) -> Tuple[bool, str, str]:
        audio_path = adir / f"{v.stem}.wav"
        ok = convert_video_to_audio(
            video_path=str(v),
            audio_path=str(audio_path),
            sr=sr,
            mono=mono,
            overwrite=overwrite,
            sanity_check_wav=sanity_check_wav,
        )
        if ok:
            return True, str(v), str(audio_path)
        return False, str(v), "conversion failed"

    pool = ThreadPool(num_workers)
    results = pool.imap_unordered(_worker, videos)

    ok_list: List[Tuple[str, str]] = []
    fail_list: List[Tuple[str, str]] = []

    for success, vpath, info in results:
        if success:
            ok_list.append((vpath, info))
            print(f"[OK] {Path(vpath).name} → {Path(info).name}")
        else:
            fail_list.append((vpath, info))
            print(f"[FAIL] {Path(vpath).name}")

    pool.close()
    pool.join()

    return {
        "total": len(videos),
        "ok_count": len(ok_list),
        "fail_count": len(fail_list),
        "ok": ok_list,
        "failed": fail_list,
    }


if __name__ == "__main__":
    video_dir = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/video_files/LAV_DF/video"
    audio_dir = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/video_files/LAV_DF/audio"
    summary = convert_video_dir_to_audio_parallel(
        video_dir=video_dir,
        audio_dir=audio_dir,
        num_workers=4,
    )

    print(summary)
