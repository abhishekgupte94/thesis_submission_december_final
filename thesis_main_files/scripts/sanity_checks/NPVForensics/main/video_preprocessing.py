#!/usr/bin/env python
"""
sanity_video_facecrops_segmentlocal_from_json.py

- JSON is loaded HERE (sanity script).
- JSON format: { "<num>": [[s,e],[s,e],...], ... }
- Video filename format: "video_<num>.mp4" (we extract <num>)
- We pass the JSON value list directly into the preprocessor method.

Includes:
- MemoryGuard clamp
- Total time measurement
"""

from pathlib import Path
import sys
import time
import json
import re

from utils.memory_guard.memory_guard import MemoryGuard

try:
    from scripts.preprocessing.video.VideoPreprocessorNPV import VideoPreprocessorNPV, VideoPreprocessorConfig  # type: ignore
except Exception:
    from VideoPreprocessorNPV import VideoPreprocessorNPV, VideoPreprocessorConfig  # type: ignore


# ===================== USER CONFIG (EDIT THESE) =====================
VIDEO_PATH = Path("/ABS/PATH/TO/video_123.mp4")

# One JSON containing all timestamp lists
TIMESTAMPS_JSON = Path("/ABS/PATH/TO/all_timestamps.json")

OUT_DIR = Path("/ABS/PATH/TO/output_facecrops_dir")

TARGET_CLIP_DURATION_SEC = 2.0
MIN_FACTOR = 0.5
MAX_FACTOR = 1.5
KEEP_FULL_WHEN_NO_FACE = True

USE_GPU = True
CTX_ID = 0

# Memory safety (Mac)
MAX_PROCESS_GB = 8.0
MIN_SYSTEM_GB = 2.0
# =======================================================


def _assert_exists(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"[SANITY] {label} not found: {p}")


def _extract_num_from_video_stem(stem: str) -> str:
    """
    Extract <num> from "video_<num>".
    Fallback: last digit group in the stem.
    """
    if stem.startswith("video_"):
        return stem.split("video_", 1)[1]

    m = re.findall(r"\d+", stem)
    if m:
        return m[-1]

    # last resort: use stem itself
    return stem


def main() -> None:
    print("\n=== [SANITY] Face crops (segmentlocal) from JSON values ===")
    total_t0 = time.time()

    _assert_exists(VIDEO_PATH, "VIDEO_PATH")
    _assert_exists(TIMESTAMPS_JSON, "TIMESTAMPS_JSON")

    guard = MemoryGuard(
        max_process_gb=MAX_PROCESS_GB,
        min_system_available_gb=MIN_SYSTEM_GB,
        throws=True,
    )
    guard.check()

    # Load JSON here (as requested)
    with TIMESTAMPS_JSON.open("r", encoding="utf-8") as f:
        ts_map = json.load(f)

    video_num = _extract_num_from_video_stem(VIDEO_PATH.stem)

    if video_num not in ts_map:
        sample_keys = list(ts_map.keys())[:15]
        raise KeyError(
            f"[SANITY] JSON key '{video_num}' not found. "
            f"Video stem='{VIDEO_PATH.stem}'. Sample keys={sample_keys}"
        )

    word_times = ts_map[video_num]  # <-- values are [[s,e], ...] as you described

    cfg = VideoPreprocessorConfig(
        insightface_model_name="buffalo_l",
        detector_size=(640, 640),
        crop_resize=(240, 240),
        target_clip_duration_sec=TARGET_CLIP_DURATION_SEC,
        use_gpu_if_available=USE_GPU,
        ctx_id=CTX_ID,
    )

    vp = VideoPreprocessorNPV(cfg=cfg)

    print(f"[SANITY] Video: {VIDEO_PATH}")
    print(f"[SANITY] JSON:  {TIMESTAMPS_JSON}")
    print(f"[SANITY] Key:   '{video_num}' (from '{VIDEO_PATH.stem}')")
    print(f"[SANITY] OUT:   {OUT_DIR}")
    print(f"[SANITY] times: {len(word_times)} entries")
    print(f"[SANITY] keep_full_when_no_face={KEEP_FULL_WHEN_NO_FACE}")

    nseg, nsaved = vp.process_and_save_facecrops_to_disk_from_word_times_segmentlocal(
        video_path=VIDEO_PATH,
        word_times=word_times,
        out_dir=OUT_DIR,
        keep_full_when_no_face=KEEP_FULL_WHEN_NO_FACE,
        min_factor=MIN_FACTOR,
        max_factor=MAX_FACTOR,
        target_clip_duration=TARGET_CLIP_DURATION_SEC,
        jpeg_quality=95,
    )

    guard.check()
    total_dt = time.time() - total_t0

    print("\n--- [SANITY] RESULT ---")
    print(f"[SANITY] Segments:        {nseg}")
    print(f"[SANITY] Frames saved:    {nsaved}")
    print(f"[SANITY] TOTAL TIME (s):  {total_dt:.2f}")
    print(f"[SANITY] TOTAL TIME (m):  {total_dt/60.0:.2f}")
    print(f"[SANITY] Output dir:      {OUT_DIR / VIDEO_PATH.stem}")
    print("=== [SANITY] DONE ===\n")


if __name__ == "__main__":
    try:
        main()
    except MemoryError as e:
        print("\n[MEMORY GUARD TRIGGERED]")
        print(str(e))
        sys.exit(2)
    except Exception as e:
        print(f"\n[SANITY] FAILED: {repr(e)}\n")
        sys.exit(1)
