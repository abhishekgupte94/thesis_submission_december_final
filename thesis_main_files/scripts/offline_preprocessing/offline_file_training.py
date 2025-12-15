#!/usr/bin/env python
"""
offline_export_avspeech.py

Stage-2 offline exporter for AVSpeech-style data.

JSON-based timestamps + optional multiprocessing.

GPU-throughput (recommended)
----------------------------
For an 8×A100 box, best practice is:
- run 8 separate processes
- pin each process to one GPU via CUDA_VISIBLE_DEVICES
- keep num_workers=1 inside each process (avoid ORT contention + huge init overhead)

Example:
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python offline_export_avspeech.py ... --shard-id $i --num-shards 8 --num-workers 1 &
done
wait
"""
from __future__ import annotations

import argparse
import json
import os  # [ADDED] for CUDA_VISIBLE_DEVICES sanity print
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, List

from audio_preprocessor_npv import AudioPreprocessorNPV
from VideoPreprocessorNPV import VideoPreprocessorNPV


# ----------------------------------------------------------------------
# Default root for offline training artifacts
# ----------------------------------------------------------------------
DEFAULT_OFFLINE_ROOT = "data/processed/AVSpeech/AVSpeech_offline_training_files"


# ----------------------------------------------------------------------
# Config for a single offline export run/batch
# ----------------------------------------------------------------------
@dataclass
class OfflineExportConfig:
    audio_root: Path
    video_root: Path
    timestamps_json: Path
    offline_root: Path
    batch_name: str

    # [ADDED] Shard context so logs/index can be made shard-safe
    shard_id: int = 0
    num_shards: int = 1

    @property
    def batch_dir(self) -> Path:
        return self.offline_root / self.batch_name

    @property
    def audio_out_dir(self) -> Path:
        # NOTE: You said your preprocessors will save into shard dirs.
        # This makes it explicit and collision-proof.
        return self.batch_dir / "audio" / f"shard_{self.shard_id}"

    @property
    def video_out_dir(self) -> Path:
        return self.batch_dir / "video" / f"shard_{self.shard_id}"

    @property
    def logs_dir(self) -> Path:
        # [MODIFIED] shard-specific logs directory to avoid concurrent write collisions
        return self.batch_dir / "logs" / f"shard_{self.shard_id}"

    @property
    def audio_log_csv(self) -> Path:
        return self.logs_dir / "audio_export_log.csv"

    @property
    def video_log_csv(self) -> Path:
        return self.logs_dir / "video_export_log.csv"

    @property
    def index_json(self) -> Path:
        # [MODIFIED] shard-specific index file (prevents “last writer wins”)
        return self.batch_dir / f"av_index_shard_{self.shard_id}.json"


# ----------------------------------------------------------------------
# Directory setup
# ----------------------------------------------------------------------
def ensure_dirs(cfg: OfflineExportConfig) -> None:
    cfg.batch_dir.mkdir(parents=True, exist_ok=True)
    cfg.audio_out_dir.mkdir(parents=True, exist_ok=True)
    cfg.video_out_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Load JSON timestamps index
# ----------------------------------------------------------------------
def load_timestamps_index(timestamps_json: Path) -> Dict[str, List[List[float]]]:
    """Load a JSON file mapping clip_id -> list[[start, end], ...]."""
    with timestamps_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, List[List[float]]] = {}
    for clip_id, segs in data.items():
        clean: List[List[float]] = []
        if not isinstance(segs, list):
            raise ValueError(f"Timestamps for {clip_id!r} must be a list, got {type(segs)}")
        for s in segs:
            if not (isinstance(s, (list, tuple)) and len(s) == 2):
                raise ValueError(f"Timestamp {s!r} for clip {clip_id!r} is invalid.")
            st, en = float(s[0]), float(s[1])
            if en > st:
                clean.append([st, en])
            else:
                print(f"[WARN] Invalid segment for {clip_id}: start={st}, end={en}")
        if clean:
            mapping[clip_id] = clean
        else:
            print(f"[WARN] No valid segments for {clip_id}, skipping clip.")
    return mapping


# ----------------------------------------------------------------------
# Guess video path from clip_id
# ----------------------------------------------------------------------
def guess_video_path(video_root: Path, clip_id: str) -> Path:
    cand = video_root / f"{clip_id}.mp4"
    if cand.exists():
        return cand

    for ext in [".mkv", ".avi", ".mov"]:
        alt = video_root / f"{clip_id}{ext}"
        if alt.exists():
            return alt

    return cand


# ----------------------------------------------------------------------
# Path normalisation
# ----------------------------------------------------------------------
def _rel_to_offline_root(path: Path, offline_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(offline_root.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


# ----------------------------------------------------------------------
# Core per-clip processing
# ----------------------------------------------------------------------
def process_one_clip(
    clip_id: str,
    segments: List[List[float]],
    cfg: OfflineExportConfig,
    audio_prep: AudioPreprocessorNPV,
    video_prep: VideoPreprocessorNPV,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Process a single clip_id: audio + video, using list-based timestamps.
    """
    audio_path = cfg.audio_root / f"{clip_id}.wav"
    video_path = guess_video_path(cfg.video_root, clip_id)
    t0 = time.time()

    info: Dict[str, Any] = {
        "clip_id": clip_id,
        "audio_path": str(audio_path),
        "video_path": str(video_path),
        "num_segments_input": len(segments),
        # [ADDED] useful context for debugging multi-run behaviour
        "shard_id": cfg.shard_id,
        "num_shards": cfg.num_shards,
    }

    if not audio_path.exists():
        info["error"] = "missing_audio"
        info["status"] = "error"
        info["proc_time_sec"] = time.time() - t0
        return False, info

    if not video_path.exists():
        info["error"] = "missing_video"
        info["status"] = "error"
        info["proc_time_sec"] = time.time() - t0
        return False, info

    audio_pt = cfg.audio_out_dir / f"{clip_id}_audio.pt"
    video_pt = cfg.video_out_dir / f"{clip_id}_video.pt"

    # Resume-safe skip
    if audio_pt.exists() and video_pt.exists():
        info.update(
            status="skipped",
            audio_pt=str(audio_pt),
            video_pt=str(video_pt),
            proc_time_sec=time.time() - t0,
        )
        return True, info

    # AUDIO
    try:
        seg_a, words_a = audio_prep.process_and_save_from_timestamps_csv_segmentlocal(
            audio_path=audio_path,
            timestamps=segments,
            out_pt_path=audio_pt,
            log_csv_path=cfg.audio_log_csv,
            clip_id=clip_id,  # USER COMMENT: may be non-original param in your function
        )
    except Exception as e:
        info["audio_error"] = str(e)
        info["status"] = "error"
        info["proc_time_sec"] = time.time() - t0
        return False, info

    # VIDEO
    try:
        seg_v, words_v = video_prep.process_and_save_from_timestamps_list_segmentlocal(
            video_path=video_path,
            timestamps=segments,
            out_pt_path=video_pt,
            log_csv_path=cfg.video_log_csv,
            clip_id=clip_id,
        )
    except Exception as e:
        info["video_error"] = str(e)
        info["status"] = "error"
        info["proc_time_sec"] = time.time() - t0
        return False, info

    info.update(
        status="ok",
        audio_pt=str(audio_pt),
        video_pt=str(video_pt),
        num_segments_audio=seg_a,
        num_segments_video=seg_v,
        num_words_audio=words_a,
        num_words_video=words_v,
        proc_time_sec=time.time() - t0,
    )

    return True, info


# ----------------------------------------------------------------------
# Worker wrapper (for ProcessPoolExecutor)
# ----------------------------------------------------------------------
def _worker_process_one_clip(args) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Worker entrypoint.

    NOTE:
    - Multiprocess mode is NOT recommended for GPU-heavy InsightFace runs,
      unless you also pin GPUs per worker process.
    """
    clip_id, segments, cfg = args
    audio_prep = AudioPreprocessorNPV()
    video_prep = VideoPreprocessorNPV()
    success, info = process_one_clip(
        clip_id=clip_id,
        segments=segments,
        cfg=cfg,
        audio_prep=audio_prep,
        video_prep=video_prep,
    )
    return clip_id, success, info


# ----------------------------------------------------------------------
# Main entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Offline AVSpeech exporter (JSON-based).")

    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--timestamps-json", required=True)
    parser.add_argument("--offline-root", default=DEFAULT_OFFLINE_ROOT)
    parser.add_argument("--batch-name", required=True)

    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)

    parser.add_argument(
        "--env",
        type=str,
        choices=["mac", "a100"],
        default="mac",
        help="Environment hint for default worker counts / tuning.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes for multiprocessing. "
            "If None, uses a sensible default based on --env. "
            "Set to 0 or 1 to disable multiprocessing."
        ),
    )

    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard-id out of range")

    # [ADDED] GPU pin sanity (you should see a single GPU id per process run)
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Decide worker count
    if args.num_workers is None:
        # [MODIFIED] A100 default: 1 worker (serial) to avoid massive GPU/ORT contention.
        num_workers = 2 if args.env == "mac" else 1
    else:
        num_workers = max(0, args.num_workers)

    cfg = OfflineExportConfig(
        audio_root=Path(args.audio_root),
        video_root=Path(args.video_root),
        timestamps_json=Path(args.timestamps_json),
        offline_root=Path(args.offline_root),
        batch_name=args.batch_name,
        # [ADDED] shard context
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    ensure_dirs(cfg)

    # Explicit JSON load (optional but useful for clarity/debugging)
    with cfg.timestamps_json.open("r", encoding="utf-8") as f:
        raw_json_data = json.load(f)
    print(f"[INFO] Loaded {len(raw_json_data)} clip entries from timestamps JSON")

    ts_mapping = load_timestamps_index(cfg.timestamps_json)
    items = sorted(ts_mapping.items(), key=lambda kv: kv[0])

    # Shard selection (UNCHANGED)
    shard_items = [
        (i, clip_id, segs)
        for i, (clip_id, segs) in enumerate(items)
        if i % args.num_shards == args.shard_id
    ]

    print(f"[INFO] Total clips: {len(items)}")
    print(f"[INFO] Shard {args.shard_id + 1}/{args.num_shards}: {len(shard_items)} clips")
    print(f"[INFO] Env: {args.env}, num_workers={num_workers}")

    # Load existing shard index if resuming
    if cfg.index_json.exists():
        with cfg.index_json.open("r", encoding="utf-8") as f:
            av_index = json.load(f)
    else:
        av_index = {}

    processed = 0

    # ------------------------------------------------------------------
    # SERIAL MODE (no multiprocessing)
    # ------------------------------------------------------------------
    if num_workers <= 1:
        print("[INFO] Running in SERIAL mode (no multiprocessing).")
        audio_prep = AudioPreprocessorNPV()
        video_prep = VideoPreprocessorNPV()

        for _, clip_id, segments in shard_items:
            success, info = process_one_clip(
                clip_id=clip_id,
                segments=segments,
                cfg=cfg,
                audio_prep=audio_prep,
                video_prep=video_prep,
            )

            av_index[clip_id] = info

            if success:
                if "audio_pt" in info:
                    av_index[clip_id]["audio_pt"] = _rel_to_offline_root(
                        Path(info["audio_pt"]), cfg.offline_root
                    )
                if "video_pt" in info:
                    av_index[clip_id]["video_pt"] = _rel_to_offline_root(
                        Path(info["video_pt"]), cfg.offline_root
                    )
                processed += 1

    # ------------------------------------------------------------------
    # MULTIPROCESS MODE (kept, but not recommended for GPU-heavy runs)
    # ------------------------------------------------------------------
    else:
        print("[WARN] Running in MULTIPROCESS mode. For GPU-heavy InsightFace runs, "
              "prefer num_workers=1 + one process per GPU via CUDA_VISIBLE_DEVICES.")
        tasks = [(clip_id, segments, cfg) for _, clip_id, segments in shard_items]

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            future_to_clip = {ex.submit(_worker_process_one_clip, t): t[0] for t in tasks}

            for fut in as_completed(future_to_clip):
                clip_id, success, info = fut.result()
                av_index[clip_id] = info

                if success:
                    if "audio_pt" in info:
                        av_index[clip_id]["audio_pt"] = _rel_to_offline_root(
                            Path(info["audio_pt"]), cfg.offline_root
                        )
                    if "video_pt" in info:
                        av_index[clip_id]["video_pt"] = _rel_to_offline_root(
                            Path(info["video_pt"]), cfg.offline_root
                        )
                    processed += 1

    # Save shard index
    with cfg.index_json.open("w", encoding="utf-8") as f:
        json.dump(av_index, f, indent=2)

    print(f"[DONE] Processed {processed} clips. Index saved to {cfg.index_json}")


if __name__ == "__main__":
    main()




# BATCH=batch_001
#
# for i in 0 1 2 3 4 5 6 7; do
#   CUDA_VISIBLE_DEVICES=$i \
#   python offline_export_avspeech.py \
#     --audio-root /data/processed/video_files/AVSpeech/audio \
#     --video-root /data/processed/video_files/AVSpeech/videos \
#     --timestamps-json /data/processed/AVSpeech/AVSpeech_timestamps/${BATCH}_segments.json \
#     --offline-root /data/processed/AVSpeech/AVSpeech_offline_training_files \
#     --batch-name $BATCH \
#     --env a100 \
#     --num-workers 1 \
#     --shard-id $i \
#     --num-shards 8 &
# done
# wait



# Ultra-quick smoke test (single GPU, 1 shard)
# Run this first if you want to sanity-check everything:
# CUDA_VISIBLE_DEVICES=0 \
# python offline_export_avspeech.py \
#   --audio-root /data/processed/video_files/AVSpeech/audio \
#   --video-root /data/processed/video_files/AVSpeech/videos \
#   --timestamps-json /data/processed/AVSpeech/AVSpeech_timestamps/batch_001_segments.json \
#   --offline-root /data/processed/AVSpeech/AVSpeech_offline_training_files \
#   --batch-name batch_001 \
#   --env a100 \
#   --num-workers 1 \
#   --shard-id 0 \
#   --num-shards 8

# In another terminal:
# watch -n 1 nvidia-smi
# You should see 8 GPUs ~independently active, not one GPU maxed and others idle.