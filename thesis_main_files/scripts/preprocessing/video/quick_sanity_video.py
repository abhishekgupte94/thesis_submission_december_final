#!/usr/bin/env python3
"""
SANITY SCRIPT
-------------
Purpose:
- Run VideoPreprocessorNPV on 1–2 videos
- Use dummy word timestamps (~5s total)
- Verify that ONE .mp4 is saved per segment
- Do NOT touch config / arguments / semantics
"""

import time
from pathlib import Path

from VideoPreprocessorNPV import (
    VideoPreprocessorNPV,
    VideoPreprocessorConfig,
)

# ------------------------------------------------------------------
# USER PATHS (EDIT THESE)
# ------------------------------------------------------------------
VIDEO_DIR = Path("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/video_files/AVSpeech/sanity_files/video")      # directory with input videos
OUT_DIR   = Path("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/temp_files/videopreprocessor_sanity_dump")      # where seg_XXXX/*.mp4 go
VIDEO_IDS = [
    "video_1.mp4",
    "video_2.mp4",  # optional second video
]

# ------------------------------------------------------------------
# Dummy word timestamps (~5 seconds total)
# Format: List[List[start_sec, end_sec]]
# ------------------------------------------------------------------
DUMMY_WORD_TIMES = [
    [0.00, 0.80],
    [0.85, 1.60],
    [1.70, 2.50],
    [2.60, 3.40],
    [3.50, 4.20],
    [4.30, 5.00],
]

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # IMPORTANT:
    # We do NOT change config semantics.
    # If InsightFace is unstable on Mac, override ONLY at runtime:
    cfg = VideoPreprocessorConfig(
        ctx_id=-1,                  # CPU on Mac
        use_gpu_if_available=False, # do NOT force CUDA
    )

    vp = VideoPreprocessorNPV(cfg)

    t_global_start = time.time()

    for vid_name in VIDEO_IDS:
        video_path = VIDEO_DIR / vid_name
        assert video_path.exists(), f"Missing video: {video_path}"

        print(f"\n=== Processing {video_path.name} ===")
        t0 = time.time()

        num_segments, total_saved = (
            vp.process_and_save_facecrops_to_disk_from_word_times_segmentlocal(
                video_path=video_path,
                word_times=DUMMY_WORD_TIMES,
                out_dir=OUT_DIR,
                keep_full_when_no_face=True,
                min_factor=0.5,
                max_factor=1.5,
                target_clip_duration=2.0,
                # jpeg_quality kept for API compatibility (ignored for video)
                jpeg_quality=95,
                out_pt_path=OUT_DIR / f"{video_path.stem}_segments.pt",
            )
        )

        dt = time.time() - t0
        print(f"Segments created      : {num_segments}")
        print(f"Frames written (total): {total_saved}")
        print(f"Time taken            : {dt:.2f} sec")

    print(f"\nTOTAL WALL TIME: {time.time() - t_global_start:.2f} sec")


if __name__ == "__main__":
    main()
