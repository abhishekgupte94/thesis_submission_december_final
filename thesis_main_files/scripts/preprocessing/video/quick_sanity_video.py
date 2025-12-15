# sanity_video_preprocessor_npv.py

from __future__ import annotations

from pathlib import Path
import time
import torch

from VideoPreprocessorNPV import VideoPreprocessorNPV, VideoPreprocessorConfig


def main() -> None:
    t0 = time.time()

    # ----------------------------
    # EDIT THESE PATHS
    # ----------------------------
    video_path = Path("/ABS/PATH/TO/YOUR_VIDEO.mp4")
    out_pt_path = Path("/ABS/PATH/TO/OUTPUT/video_segments.pt")

    # Optional: also save crops to disk to visually confirm face-cropping
    out_crops_dir = Path("/ABS/PATH/TO/OUTPUT/debug_face_crops/")

    # In-memory timestamps (seconds)
    # Format: [[start_sec, end_sec], ...]
    word_times = [
        [0.10, 0.35],
        [0.40, 0.80],
        [1.20, 1.55],
        [1.60, 2.05],
    ]

    # ----------------------------
    # Run
    # ----------------------------
    cfg = VideoPreprocessorConfig(
        crop_resize=(240, 240),
        target_clip_duration_sec=2.0,
        use_gpu_if_available=True,
        ctx_id=0,
    )
    proc = VideoPreprocessorNPV(cfg)

    num_segments, num_words = proc.process_and_save_from_timestamps_csv_segmentlocal(
        video_path=video_path,
        word_times=word_times,
        out_pt_path=out_pt_path,
        keep_full_when_no_face=True,
    )

    # Extra: dump crops to disk for visual sanity
    segs2, total_saved = proc.process_and_save_facecrops_to_disk_from_word_times_segmentlocal(
        video_path=video_path,
        word_times=word_times,
        out_dir=out_crops_dir,
        keep_full_when_no_face=True,
    )

    # ----------------------------
    # Validate saved payload
    # ----------------------------
    payload = torch.load(out_pt_path, map_location="cpu")

    required = [
        "video_file",
        "video_segments",
        "num_segments",
        "num_words",
        "timestamps_csv",
        "pt_rel_path",
        "config",
    ]
    for k in required:
        assert k in payload, f"Missing key in payload: {k}"

    assert payload["num_words"] == num_words
    assert payload["num_segments"] == num_segments

    video_segments = payload["video_segments"]
    assert isinstance(video_segments, list), "video_segments should be a list"
    assert len(video_segments) == num_segments

    # Each element is a Tensor shaped (F, 3, H, W) or empty (0,3,0,0)
    for i, seg in enumerate(video_segments):
        assert isinstance(seg, torch.Tensor), f"video_segments[{i}] not a Tensor"
        assert seg.ndim == 4, f"video_segments[{i}] expected 4D (F,C,H,W), got {tuple(seg.shape)}"
        assert seg.shape[1] == 3, f"video_segments[{i}] expected C=3, got {seg.shape[1]}"
        assert seg.dtype in (torch.float32, torch.float16), f"video_segments[{i}] unexpected dtype {seg.dtype}"

    dt = time.time() - t0
    print("=== [SANITY] VideoPreprocessorNPV ===")
    print(f"video: {video_path}")
    print(f"saved_pt: {out_pt_path}")
    print(f"saved_crops_dir: {out_crops_dir}")
    print(f"num_words={num_words} num_segments={num_segments}")
    print(f"disk_crops: segs={segs2} total_frames_saved={total_saved}")
    print(f"elapsed_sec={dt:.2f}")
    print("OK")


if __name__ == "__main__":
    main()
