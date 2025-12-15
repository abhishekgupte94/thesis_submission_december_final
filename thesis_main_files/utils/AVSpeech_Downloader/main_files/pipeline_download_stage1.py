#!/usr/bin/env python3
"""
pipeline_download_stage1.py

Stage-1 AVSpeech pipeline with TWO stages:

1) VIDEO STAGE (download + trim in parallel from YouTube stream URL)
2) AUDIO STAGE (extract raw audio in parallel from trimmed local videos)

ASSUMPTION
----------
The input AVSpeech CSV is already "clean" and has the columns:

    serial_id, yt_id, start_time, end_time, x, y

where:
    - serial_id : unique integer identifier (0..N-1)
    - yt_id     : YouTube video ID
    - start_time, end_time : segment boundaries in seconds
    - x, y      : coordinates (not used in processing, but preserved)

PIPELINE OVERVIEW
-----------------
VIDEO STAGE (mode: "videos" or "all"):

    - Load jobs from the CSV.
    - Skip entries that:
         * already exist in stage1_video_success.csv
         * or have serial_id listed in stage1_video_bad.txt
         * or have yt_id with previous yt_dlp errors
    - For each NEW job (optionally capped by --max-videos):
         * Use yt_dlp with download=False to get a direct media URL
           for the YouTube video.
         * Use ffmpeg on that URL to trim [start_time, end_time] to:
               <video_dir>/video_<serial_id>.mp4
         * On success → append row to stage1_video_success.csv
               (serial_id, yt_id, start_time, end_time, video_path)
         * On failure → append line to stage1_video_bad.txt
               "serial_id, yt_id, ERROR (yt_dlp)" or "ERROR (ffmpeg video)"
    - All jobs are executed in parallel with ThreadPool(imap_unordered).

AUDIO STAGE (mode: "audio" or "all"):

    - Build jobs only for serial_ids that:
         * appear in stage1_video_success.csv
         * but not in stage1_audio_success.csv
         * and not in stage1_audio_bad.txt
    - For each job:
         * Run ffmpeg on:
               <video_dir>/video_<serial_id>.mp4
           to extract 16 kHz mono WAV:
               <audio_dir>/trim_audio_train<serial_id>.wav
         * On success → append row to stage1_audio_success.csv
               (serial_id, yt_id, audio_path)
         * On failure → append line to stage1_audio_bad.txt
               "serial_id, yt_id, ERROR (missing video / ffmpeg audio)"
    - Again, executed in parallel via ThreadPool.
"""

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import List, Optional, Set, Tuple

import pandas as pd
from yt_dlp  import YoutubeDL


# -----------------------------
# Default paths & constants
# -----------------------------

DEFAULT_VIDEO_DIR = (
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/data/processed/video_files/AVSpeech/video"
)
DEFAULT_AUDIO_DIR = (
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/data/processed/video_files/AVSpeech/audio"
)
DEFAULT_SR = 16000
DEFAULT_DURATION = 7.0
DEFAULT_VIDEO_FORMAT = "bv*[height<=360]+ba/b[height<=360]"  # small, fast streams

# Video stage bookkeeping
DEFAULT_VIDEO_SUCCESS_CSV = (
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/temp_files/AVSpeech/stage1_video_success.csv"
)
DEFAULT_VIDEO_BAD_LOG = (
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/temp_files/AVSpeech/stage1_video_bad.txt"
)

# Audio stage bookkeeping
DEFAULT_AUDIO_SUCCESS_CSV = (
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/temp_files/AVSpeech/stage1_audio_success.csv"
)
DEFAULT_AUDIO_BAD_LOG = (
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/temp_files/AVSpeech/stage1_audio_bad.txt"
)


@dataclass
class SampleJob:
    """
    Represents a single AVSpeech entry pulled from the CLEAN CSV.

    serial_id  : unique ID used for naming and logging.
    yt_id      : YouTube ID.
    start_time : clip start time (seconds).
    end_time   : clip end time (seconds).
    x, y       : metadata (not used in processing here).
    """
    serial_id: int
    yt_id: str
    start_time: float
    end_time: float
    x: float
    y: float


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ============================================================================
# Helper: load success / bad states for resume logic
# ============================================================================

def load_previous_video_success_serials(success_csv_path: str) -> Set[int]:
    """Return set of serial_ids already logged as VIDEO successes."""
    if not os.path.exists(success_csv_path):
        return set()
    try:
        df = pd.read_csv(success_csv_path)
    except Exception:
        return set()
    if "serial_id" not in df.columns:
        return set()
    serials: Set[int] = set()
    for _, row in df.iterrows():
        try:
            serials.add(int(row["serial_id"]))
        except Exception:
            continue
    return serials


def load_previous_audio_success_serials(success_csv_path: str) -> Set[int]:
    """Return set of serial_ids already logged as AUDIO successes."""
    if not os.path.exists(success_csv_path):
        return set()
    try:
        df = pd.read_csv(success_csv_path)
    except Exception:
        return set()
    if "serial_id" not in df.columns:
        return set()
    serials: Set[int] = set()
    for _, row in df.iterrows():
        try:
            serials.add(int(row["serial_id"]))
        except Exception:
            continue
    return serials


def load_bad_serials_from_log(bad_log_path: str) -> Set[int]:
    """
    Return set of serial_ids recorded in a BAD log.

    Each line is expected to start with:
        serial_id, yt_id, ERROR (...)
    """
    if not os.path.exists(bad_log_path):
        return set()
    serials: Set[int] = set()
    with open(bad_log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue
            try:
                sid = int(parts[0])
                serials.add(sid)
            except ValueError:
                continue
    return serials


def load_unavailable_yt_ids(bad_log_path: str) -> Set[str]:
    """
    Return set of yt_ids that previously produced yt_dlp errors.

    We look for lines like:
        "serial_id, yt_id, ERROR (yt_dlp)"
    """
    if not os.path.exists(bad_log_path):
        return set()
    ids: Set[str] = set()
    with open(bad_log_path, "r") as f:
        for line in f:
            line = line.strip()
            if "ERROR (yt_dlp)" not in line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            yt_id = parts[1]
            if yt_id:
                ids.add(yt_id)
    return ids


# ============================================================================
# CSV append helpers (consistent paths included)
# ============================================================================

def append_video_success_row(
    success_csv_path: str,
    job: SampleJob,
    video_dir: str,
) -> None:
    """
    Append a VIDEO success row:

        serial_id, yt_id, start_time, end_time, video_path
    """
    ensure_header = not os.path.exists(success_csv_path)
    video_path = os.path.join(video_dir, f"video_{job.serial_id}.mp4")

    with open(success_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if ensure_header:
            writer.writerow(
                ["serial_id", "yt_id", "start_time", "end_time", "video_path"]
            )
        writer.writerow(
            [job.serial_id, job.yt_id, job.start_time, job.end_time, video_path]
        )


def append_audio_success_row(
    success_csv_path: str,
    serial_id: int,
    yt_id: str,
    audio_dir: str,
) -> None:
    """
    Append an AUDIO success row:

        serial_id, yt_id, audio_path
    """
    ensure_header = not os.path.exists(success_csv_path)
    audio_path = os.path.join(audio_dir, f"trim_audio_train{serial_id}.wav")

    with open(success_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if ensure_header:
            writer.writerow(["serial_id", "yt_id", "audio_path"])
        writer.writerow([serial_id, yt_id, audio_path])


# ============================================================================
# Job loading from CLEAN CSV
# ============================================================================

def load_video_jobs_from_clean_csv(
    csv_path: str,
    max_videos: Optional[int],
    forced_duration: float,
    use_csv_end: bool,
    existing_video_serials: Set[int],
    unavailable_yt_ids: Set[str],
    bad_serials: Set[int],
) -> List[SampleJob]:
    """
    Build list of VIDEO jobs from the cleaned CSV.

    Skips:
        - serial_id already in existing_video_serials
        - serial_id already in bad_serials
        - yt_id in unavailable_yt_ids
    """
    df = pd.read_csv(csv_path)
    required = {"serial_id", "yt_id", "start_time", "end_time"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    jobs: List[SampleJob] = []

    for _, row in df.iterrows():
        try:
            serial_id = int(row["serial_id"])
            yt_id = str(row["yt_id"])
            start_raw = float(row["start_time"])
            csv_end = float(row["end_time"])
        except Exception:
            continue

        if serial_id in existing_video_serials or serial_id in bad_serials:
            continue
        if yt_id in unavailable_yt_ids:
            continue

        try:
            x = float(row["x"])
            y = float(row["y"])
        except Exception:
            x, y = 0.0, 0.0

        # Duration clamping logic identical to previous pipeline
        if use_csv_end and csv_end is not None:
            raw_end = csv_end
        else:
            raw_end = start_raw + forced_duration

        duration = raw_end - start_raw
        if duration <= 0:
            duration = forced_duration
        elif duration > forced_duration:
            duration = forced_duration

        end_time = start_raw + duration

        jobs.append(
            SampleJob(
                serial_id=serial_id,
                yt_id=yt_id,
                start_time=start_raw,
                end_time=end_time,
                x=x,
                y=y,
            )
        )

        if max_videos is not None and len(jobs) >= max_videos:
            break

    return jobs


def load_audio_jobs_from_video_success(
    video_success_csv: str,
    audio_success_csv: str,
    audio_bad_log: str,
) -> List[Tuple[int, str]]:
    """
    Build AUDIO jobs (serial_id, yt_id) based on:

        - stage1_video_success.csv
        - skipping serial_ids in audio_success and audio_bad_log
    """
    if not os.path.exists(video_success_csv):
        print(f"[AUDIO] Video success CSV not found: {video_success_csv}")
        return []

    df = pd.read_csv(video_success_csv)
    if not {"serial_id", "yt_id"}.issubset(df.columns):
        print("[AUDIO] Video success CSV missing 'serial_id' or 'yt_id'.")
        return []

    done_audio = load_previous_audio_success_serials(audio_success_csv)
    bad_audio = load_bad_serials_from_log(audio_bad_log)

    jobs: List[Tuple[int, str]] = []
    for _, row in df.iterrows():
        try:
            sid = int(row["serial_id"])
            yt_id = str(row["yt_id"])
        except Exception:
            continue
        if sid in done_audio or sid in bad_audio:
            continue
        jobs.append((sid, yt_id))
    return jobs


# ============================================================================
# yt_dlp + ffmpeg helpers (VidInfo-style download logic)
# ============================================================================

def get_stream_url(
    yt_id: str,
    video_format: str,
) -> Optional[str]:
    """
    Use yt_dlp with download=False to get a direct media URL for a YouTube ID.

    This is equivalent to your VidInfo-based code, but returns just the URL.

    Returns:
        stream_url (str) on success, or None on error.
    """
    yt_url = f"https://www.youtube.com/watch?v={yt_id}"

    ydl_opts = {
        "format": video_format,
        "quiet": True,
        "ignoreerrors": True,
        "no_warnings": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url=yt_url, download=False)
            if not info or "url" not in info:
                return None
            return info["url"]
    except Exception as e:
        print(f"[yt_dlp] ERROR for {yt_id}: {e}")
        return None


def run_ffmpeg(cmd: List[str]) -> bool:
    """
    Run ffmpeg command and return True on success, False on non-zero exit.
    """
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


# ============================================================================
# VIDEO STAGE
# ============================================================================

def process_video_sample(
    job: SampleJob,
    video_dir: str,
    video_format: str,
) -> Tuple[SampleJob, str]:
    """
    VIDEO worker using your VidInfo-like logic:

        1) Get stream URL via yt_dlp (download=False).
        2) Call ffmpeg with -ss/-to directly on that URL.
        3) Save as video_<serial_id>.mp4 in video_dir.

    Returns:
        (job, status_string)
    """
    serial_id = job.serial_id
    yt_id = job.yt_id

    print(
        f"[VIDEO JOB {serial_id}] yt_id={yt_id}, "
        f"{job.start_time:.2f}s -> {job.end_time:.2f}s"
    )

    # Get direct media URL from YouTube
    stream_url = get_stream_url(yt_id, video_format=video_format)
    if stream_url is None:
        # Robust yt_dlp error handling, won't crash the pool
        return job, f"{serial_id}, {yt_id}, ERROR (yt_dlp)"

    ensure_dir(video_dir)
    video_out = os.path.join(video_dir, f"video_{serial_id}.mp4")

    # Use ffmpeg to trim the remote stream directly
    # (keeps your original encoding settings)
    ffmpeg_cmd = [
        "ffmpeg",
        "-ss", str(job.start_time),
        "-to", str(job.end_time),
        "-i", stream_url,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        # Previously: "-c:a", "copy",
        # Better: re-encode to something standard (e.g., AAC)
        "-c:a", "aac",
        "-b:a", "128k",
        "-y", video_out,
    ]

    if not run_ffmpeg(ffmpeg_cmd):
        # ffmpeg error handled locally; we just mark this as bad
        return job, f"{serial_id}, {yt_id}, ERROR (ffmpeg video)"

    return job, f"{yt_id}, DONE_VIDEO!"


def process_video_jobs_in_parallel(
    jobs: List[SampleJob],
    video_dir: str,
    video_format: str,
    num_workers: int,
    video_success_csv: str,
    video_bad_log: str,
) -> None:
    """
    Run VIDEO jobs in parallel with ThreadPool.imap_unordered.

    - On success → append to video_success_csv.
    - On failure → append a human-readable line to video_bad_log.
    """

    def _worker(job: SampleJob) -> Tuple[SampleJob, str]:
        return process_video_sample(job, video_dir=video_dir, video_format=video_format)

    pool = ThreadPool(num_workers)
    results = pool.imap_unordered(_worker, jobs)

    bad_lines: List[str] = []

    for job, status in results:
        print("[VIDEO STATUS]", status)
        if "DONE_VIDEO!" in status:
            append_video_success_row(video_success_csv, job, video_dir)
        else:
            bad_lines.append(status)

    pool.close()
    pool.join()

    if bad_lines:
        with open(video_bad_log, "a") as f:
            for line in bad_lines:
                f.write(line + "\n")
        print(f"[VIDEO PIPELINE] {len(bad_lines)} errors logged to {video_bad_log}")
    else:
        print("[VIDEO PIPELINE] All video jobs completed successfully.")


# ============================================================================
# AUDIO STAGE
# ============================================================================

def process_audio_sample(
    serial_id: int,
    yt_id: str,
    video_dir: str,
    audio_dir: str,
    sr: int,
) -> Tuple[int, str]:
    """
    AUDIO worker:

        1) Read local video_<serial_id>.mp4
        2) Extract mono 16 kHz PCM WAV via ffmpeg to trim_audio_train<serial_id>.wav

    We make the WAV encoding explicit and then sanity-check the output.
    """
    ensure_dir(audio_dir)

    video_path = os.path.join(video_dir, f"video_{serial_id}.mp4")
    audio_path = os.path.join(audio_dir, f"trim_audio_train{serial_id}.wav")

    if not os.path.exists(video_path):
        return serial_id, f"{serial_id}, {yt_id}, ERROR (missing video)"

    # Explicit WAV settings + audio codec
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",                      # overwrite output if it exists
        "-i", video_path,          # input video
        "-vn",                     # drop video stream
        "-acodec", "pcm_s16le",    # 16-bit PCM (standard WAV)
        "-ar", str(sr),            # sample rate (e.g. 16000)
        "-ac", "1",                # mono
        audio_path,                # output .wav
    ]

    if not run_ffmpeg(ffmpeg_cmd):
        return serial_id, f"{serial_id}, {yt_id}, ERROR (ffmpeg audio)"

    # --- Extra safety: check that the WAV is not empty/corrupt ---
    try:
        # 44 bytes is the minimum valid WAV header size.
        if (not os.path.exists(audio_path)) or (os.path.getsize(audio_path) < 44):
            return serial_id, f"{serial_id}, {yt_id}, ERROR (empty/invalid wav)"
    except OSError:
        return serial_id, f"{serial_id}, {yt_id}, ERROR (wav filesize check)"

    return serial_id, f"{yt_id}, DONE_AUDIO!"



def process_audio_jobs_in_parallel(
    jobs: List[Tuple[int, str]],
    video_dir: str,
    audio_dir: str,
    sr: int,
    num_workers: int,
    audio_success_csv: str,
    audio_bad_log: str,
) -> None:
    """
    Run AUDIO jobs in parallel with ThreadPool.imap_unordered.

    - On success → append to audio_success_csv.
    - On failure → append line to audio_bad_log.
    """

    def _worker(job: Tuple[int, str]) -> Tuple[int, str]:
        serial_id, yt_id = job
        return process_audio_sample(
            serial_id=serial_id,
            yt_id=yt_id,
            video_dir=video_dir,
            audio_dir=audio_dir,
            sr=sr,
        )

    pool = ThreadPool(num_workers)
    results = pool.imap_unordered(_worker, jobs)

    bad_lines: List[str] = []

    for serial_id, status in results:
        print("[AUDIO STATUS]", status)
        if "DONE_AUDIO!" in status:
            yt_id = status.split(",")[0].strip()
            append_audio_success_row(audio_success_csv, serial_id, yt_id, audio_dir)
        else:
            bad_lines.append(status)

    pool.close()
    pool.join()

    if bad_lines:
        with open(audio_bad_log, "a") as f:
            for line in bad_lines:
                f.write(line + "\n")
        print(f"[AUDIO PIPELINE] {len(bad_lines)} errors logged to {audio_bad_log}")
    else:
        print("[AUDIO PIPELINE] All audio jobs completed successfully.")


# ============================================================================
# CLI + main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage-1 AVSpeech downloader/extractor:\n"
            "- VIDEO: yt_dlp (download=False) + ffmpeg trim from stream URL\n"
            "- AUDIO: ffmpeg audio extraction from local trimmed videos\n"
            "- Clean CSV assumed: serial_id, yt_id, start_time, end_time, x, y"
        )
    )

    parser.add_argument(
        "--csv",
        # required=True,
        default="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/raw/csv/AVSpeech/avspeech_train.csv",
        help="Path to CLEAN AVSpeech CSV (serial_id, yt_id, start_time, end_time, x, y).",
    )
    parser.add_argument(
        "--mode",
        choices=["videos", "audio", "all"],
        default="all",
        help="Which stage(s) to run: videos / audio / all.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Cap on number of NEW video jobs in this run (VIDEO STAGE).",
    )
    parser.add_argument(
        "--video-dir",
        default=DEFAULT_VIDEO_DIR,
        help="Directory for trimmed videos (video_<serial_id>.mp4).",
    )
    parser.add_argument(
        "--audio-dir",
        default=DEFAULT_AUDIO_DIR,
        help="Directory for audio files (trim_audio_train<serial_id>.wav).",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SR,
        help="Audio sampling rate (Hz) for WAV output.",
    )
    parser.add_argument(
        "--video-format",
        default=DEFAULT_VIDEO_FORMAT,
        help="yt_dlp format string (e.g. 'best', 'bv*[height<=360]+ba/b[height<=360]').",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel threads per stage.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Upper bound on clip duration (seconds); used to clamp CSV end_time.",
    )
    parser.add_argument(
        "--use-csv-end",
        action="store_true",
        help=(
            "If set, use CSV end_time (clamped to <= duration); "
            "otherwise always use start_time + duration."
        ),
    )

    # Video bookkeeping
    parser.add_argument(
        "--video-success-csv",
        default=DEFAULT_VIDEO_SUCCESS_CSV,
        help="Path to CSV logging VIDEO STAGE successes.",
    )
    parser.add_argument(
        "--video-bad-log",
        default=DEFAULT_VIDEO_BAD_LOG,
        help="Path to TXT logging VIDEO STAGE failures.",
    )

    # Audio bookkeeping
    parser.add_argument(
        "--audio-success-csv",
        default=DEFAULT_AUDIO_SUCCESS_CSV,
        help="Path to CSV logging AUDIO STAGE successes.",
    )
    parser.add_argument(
        "--audio-bad-log",
        default=DEFAULT_AUDIO_BAD_LOG,
        help="Path to TXT logging AUDIO STAGE failures.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------
    # VIDEO STAGE
    # -------------------------
    if args.mode in ("videos", "all"):
        existing_vid = load_previous_video_success_serials(args.video_success_csv)
        bad_vid = load_bad_serials_from_log(args.video_bad_log)
        bad_yt = load_unavailable_yt_ids(args.video_bad_log)

        print("[VIDEO PIPELINE] Loading jobs from:", args.csv)
        print(f"[VIDEO PIPELINE] Skipping {len(existing_vid)} already-successful serial_ids.")
        print(f"[VIDEO PIPELINE] Skipping {len(bad_vid)} known-bad serial_ids.")
        print(f"[VIDEO PIPELINE] Skipping {len(bad_yt)} yt_dlp-unavailable yt_ids.")

        video_jobs = load_video_jobs_from_clean_csv(
            args.csv,
            max_videos=args.max_videos,
            forced_duration=args.duration,
            use_csv_end=args.use_csv_end,
            existing_video_serials=existing_vid,
            unavailable_yt_ids=bad_yt,
            bad_serials=bad_vid,
        )

        print(f"[VIDEO PIPELINE] Loaded {len(video_jobs)} NEW video jobs.")

        if video_jobs:
            process_video_jobs_in_parallel(
                video_jobs,
                video_dir=args.video_dir,
                video_format=args.video_format,
                num_workers=args.num_workers,
                video_success_csv=args.video_success_csv,
                video_bad_log=args.video_bad_log,
            )
        else:
            print("[VIDEO PIPELINE] Nothing new to do.")

    # -------------------------
    # AUDIO STAGE
    # -------------------------
    if args.mode in ("audio", "all"):
        print("[AUDIO PIPELINE] Preparing audio jobs from video_success_csv.")
        audio_jobs = load_audio_jobs_from_video_success(
            video_success_csv=args.video_success_csv,
            audio_success_csv=args.audio_success_csv,
            audio_bad_log=args.audio_bad_log,
        )

        print(f"[AUDIO PIPELINE] Loaded {len(audio_jobs)} NEW audio jobs.")

        if audio_jobs:
            process_audio_jobs_in_parallel(
                audio_jobs,
                video_dir=args.video_dir,
                audio_dir=args.audio_dir,
                sr=args.sr,
                num_workers=args.num_workers,
                audio_success_csv=args.audio_success_csv,
                audio_bad_log=args.audio_bad_log,
            )
        else:
            print("[AUDIO PIPELINE] Nothing new to do.")


if __name__ == "__main__":
    main()
