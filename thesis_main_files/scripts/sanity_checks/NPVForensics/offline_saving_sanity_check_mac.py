
"""
sanity_offline_export_mac_safe.py
=================================

Mac-safe sanity check for OFFLINE face/image export.

- Iterates over a small number of source video/images.
- Uses your existing face/image extraction logic.
- Saves crops to an output directory.
- Monitors RAM and MPS GPU memory.
- Aborts if memory usage exceeds thresholds.

Plug in your own `extract_and_save_faces` or `process_and_save_image`
implementation where indicated.
"""

import os
import glob
import psutil
import torch

# TODO: replace this with your actual preprocessor import
# from scripts.
from scripts.preprocessing.main. import


# ================================================================
# 1. Device detection (CPU / MPS)
# ================================================================
def get_device():
    if torch.backends.mps.is_available():
        print("[INFO] Using Apple MPS GPU for face extraction (if supported).")
        return torch.device("mps")
    else:
        print("[INFO] Using CPU for face extraction.")
        return torch.device("cpu")


device = get_device()


# ================================================================
# 2. Memory monitoring utilities
# ================================================================
def check_ram(threshold_gb: float = 12.0):
    """Abort if system RAM usage exceeds threshold."""
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    print(f"[RAM] Used: {used_gb:.2f} GB")

    if used_gb > threshold_gb:
        raise MemoryError(
            f"RAM usage too high ({used_gb:.2f} GB > {threshold_gb} GB) — aborting."
        )


def check_mps_memory(threshold_gb: float = 8.0):
    """Monitor Apple MPS GPU memory and abort if above threshold."""
    if not torch.backends.mps.is_available():
        print("[MPS] Not available — skipping GPU memory check.")
        return

    try:
        used = torch.mps.current_allocated_memory() / (1024 ** 3)
        reserved = torch.mps.driver_allocated_memory() / (1024 ** 3)
        print(f"[MPS] Allocated: {used:.2f} GB | Reserved: {reserved:.2f} GB")

        if reserved > threshold_gb:
            raise MemoryError(
                f"MPS reserved memory too high "
                f"({reserved:.2f} GB > {threshold_gb} GB) — aborting."
            )
    except Exception as e:
        print("[MPS] Memory check unavailable:", e)


# ================================================================
# 3. Your face/image export implementation (placeholder)
# ================================================================
def extract_and_save_faces_stub(
    video_path: str,
    output_dir: str,
    max_faces_per_frame: int = 1,
) -> int:
    """
    [PLACEHOLDER] Replace with your real face/image saving logic.

    For now, just simulates processing and creates a dummy file.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    dummy_out = os.path.join(output_dir, f"{base}_dummy_face.png")

    # Simulate some work:
    # In your real function, you would:
    #   - open the video
    #   - iterate frames
    #   - detect faces
    #   - optionally move tensors to `device`
    #   - crop and save face patches
    with open(dummy_out, "wb") as f:
        f.write(b"")  # empty file to prove the path is writeable

    return 1  # pretend we saved 1 face


# ================================================================
# 4. Main sanity routine
# ================================================================
def main():
    # ---------------- Source + destination config ---------------- #
    source_dir = "raw_videos"      # TODO: change to your raw video dir
    output_root = "offline_faces"  # TODO: change to your offline save directory

    # Pattern for video; adjust extensions as needed
    pattern = os.path.join(source_dir, "*.mp4")
    video_paths = sorted(glob.glob(pattern))

    if not video_paths:
        print(f"[WARN] No video found in {source_dir} (pattern: {pattern})")
        return

    # Limit how many we process in this sanity check
    max_videos = 3
    print(f"[INFO] Found {len(video_paths)} video, processing at most {max_videos}.")

    # Initial memory check
    check_ram()
    check_mps_memory()

    total_saved_global = 0

    for i, video_path in enumerate(video_paths[:max_videos]):
        print("\n" + "=" * 60)
        print(f"[STEP] Processing video {i+1}/{max_videos}: {video_path}")
        print("=" * 60)

        # Check memory before each video
        check_ram()
        check_mps_memory()

        # Build output dir per video
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        video_out_dir = os.path.join(output_root, video_id)

        # ---- Call your real extractor here ----
        # Example if you have a GPU-enabled extractor:
        #
        #   extract_and_save_faces(
        #       video_path=video_path,
        #       output_dir=video_out_dir,
        #       max_faces_per_frame=2,
        #       device=device,
        #   )
        #
        # For now we call the stub:
        num_saved = extract_and_save_faces_stub(
            video_path=video_path,
            output_dir=video_out_dir,
            max_faces_per_frame=1,
        )

        total_saved_global += num_saved
        print(f"[INFO] Saved {num_saved} faces for this video into {video_out_dir}")

        # Check memory after each video
        check_ram()
        check_mps_memory()

    print("\n[SUMMARY] Finished offline face/image export sanity.")
    print(f"          Total faces/images saved: {total_saved_global}")
    print(f"          Output root: {output_root}")


if __name__ == "__main__":
    main()