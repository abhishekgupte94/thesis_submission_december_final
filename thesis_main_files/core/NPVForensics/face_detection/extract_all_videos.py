import os
import cv2
from pathlib import Path

# Set of video extensions considered valid.
# You can expand this depending on your dataset.
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_all_videos(input_videos_dir, output_frames_root):
    """
    Extract **every frame** from each video in `input_videos_dir`,
    and save them under:

        output_frames_root / <video_name_without_ext> / 0001.jpg

    This is a pure visual preprocessing script:
    ---------------------------------------------------------
    • Reads the video using OpenCV
    • Saves **every decoded frame** (no skipping)
    • Does **not** subsample frames
    • Does **not** select non-critical phonemes/visemes
    • Assumes input video are ~25 FPS but does NOT resample —
      it simply extracts whatever frames OpenCV decodes.

    This ensures that for a 25 FPS source video, all 25 frames
    per second will be extracted exactly as they appear.
    """

    input_dir = Path(input_videos_dir)
    output_root = Path(output_frames_root)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_videos_dir}")

    # Ensure output root directory exists
    output_root.mkdir(parents=True, exist_ok=True)

    # Collect all video with known extensions
    video_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    )

    if len(video_files) == 0:
        print("No video files found in directory:", input_videos_dir)
        return

    print(f"Found {len(video_files)} video in {input_videos_dir}.\n")

    for vid_path in video_files:
        video_name = vid_path.stem                           # Without extension
        out_dir = output_root / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n---- Extracting frames from: {vid_path.name} ----")
        print(f"Saving frames to: {out_dir}")

        # Open the video
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {vid_path}")
            continue

        # FPS check (expected ~25 for your dataset)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  Reported FPS: {fps:.2f}  (we still save *every* frame intact)")

        frame_idx = 1

        # Read all frames sequentially
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # No frame skipping — save EVERY decoded frame
            frame_path = out_dir / f"{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)

            frame_idx += 1

        cap.release()
        print(f"  -> Extracted {frame_idx - 1} frames.")

    print("\nAll video processed successfully.")


# -----------------------
# Example usage (customize)
# -----------------------
# output_frames_root = (
#     "/Users/abhishekgupte_macbookpro/PycharmPro..."
#     "ject_combined_repo_clean/thesis_main_files/scripts/sanity_checks"
# )
# input_videos_dir = "/data/raw/video_files/sanity_check_ten_videos/"
# Sample code run below
# if __name__ == "__main__":
#     extract_all_videos(
#         input_videos_dir=input_videos_dir,
#         output_frames_root=output_frames_root,
#     )
