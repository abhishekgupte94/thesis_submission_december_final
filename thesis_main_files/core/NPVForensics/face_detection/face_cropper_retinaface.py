"""
face_cropper_retinaface.py

This script processes frame folders produced by extract_all_videos.py
and applies RetinaFace detection to crop face regions.

Pipeline:
---------------------------------------------------------
1) extract_all_videos.py dumps EVERY frame (e.g., 25 FPS).
2) This script:
      • loads each frame
      • detects the largest face using RetinaFace
      • enlarges the bounding box (1.3×) for stable lip/face context
      • crops and saves into a parallel folder structure

Notes:
---------------------------------------------------------
• This script processes EVERY FRAME. No skipping.
• It does NOT implement any non-critical phoneme or viseme logic.
• All frames remain aligned with their original timestamps.
• Ideal for feeding into LFA-ST, TALL-Swin, or any lip/face encoder.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# RetinaFace from insightface
from insightface.app import FaceAnalysis


# -----------------------------
# USER CONFIGURATION
# -----------------------------
SOURCE_FRAMES_ROOT = "/path/to/raw_frames/"        # e.g., dataset/frames_raw/
TARGET_FRAMES_ROOT = "/path/to/cropped_frames/"    # e.g., dataset/frames_face/
IMG_EXTS = {".png", ".jpg", ".jpeg"}

# Enlarge the bounding box by 30%
# This is a common heuristic in lip-reading & facial models
ENLARGE_RATIO = 1.3


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def enlarge_bbox(bbox, img_w, img_h, ratio=1.3):
    """
    Enlarge a bounding box around its center by a given ratio.

    Parameters:
        bbox : (x1, y1, x2, y2)
        img_w, img_h : image dimensions
        ratio : how much to scale the box by

    Returns:
        Enlarged (x1, y1, x2, y2) clamped to image boundaries.
    """
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    new_w = w * ratio
    new_h = h * ratio

    nx1 = int(round(cx - new_w / 2.0))
    ny1 = int(round(cy - new_h / 2.0))
    nx2 = int(round(cx + new_w / 2.0))
    ny2 = int(round(cy + new_h / 2.0))

    # Clamp to borders
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)

    return nx1, ny1, nx2, ny2


def detect_largest_face(detector, img):
    """
    Run RetinaFace to detect all faces, then choose the largest.

    Returns:
        bbox = (x1, y1, x2, y2)   OR   None if no face.
    """
    faces = detector.get(img)
    if len(faces) == 0:
        return None

    # Choose face with largest area
    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    return tuple(largest.bbox.astype(int))


# -----------------------------
# MAIN PROCESSING FUNCTION
# -----------------------------
def preprocess_dataset():
    """
    Crop faces for every frame stored in SOURCE_FRAMES_ROOT.

    Expected structure:
        SOURCE_FRAMES_ROOT /
            video_id_001 /
                0001.jpg
                0002.jpg
            video_id_002 /
                ...

    Output structure mirrors this in TARGET_FRAMES_ROOT.

    Behavior:
        • every frame is read
        • RetinaFace detects the largest face
        • bounding box is enlarged via ENLARGE_RATIO
        • cropped region is saved
        • if no face detected → full frame is saved
          (safer than skipping; keeps all temporal info)
    """

    # Load RetinaFace (GPU=0; set ctx_id=-1 for CPU)
    detector = FaceAnalysis(name="retinaface_r50_v1")
    detector.prepare(ctx_id=0, det_size=(640, 640))

    source_root = Path(SOURCE_FRAMES_ROOT)
    target_root = Path(TARGET_FRAMES_ROOT)
    target_root.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        raise FileNotFoundError(f"Source frames root does not exist: {SOURCE_FRAMES_ROOT}")

    # All subdirectories are treated as video folders
    video_folders = sorted([p for p in source_root.iterdir() if p.is_dir()])

    print(f"Found {len(video_folders)} video folders.\n")

    for video_dir in tqdm(video_folders, desc="Videos"):
        rel_path = video_dir.relative_to(source_root)
        out_dir = target_root / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect frames inside this directory
        frame_files = sorted(
            [p for p in video_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        )

        for frame_path in frame_files:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue  # corrupted frame or unreadable

            h, w = img.shape[:2]

            # detect face
            bbox = detect_largest_face(detector, img)

            # If no face: keep original frame
            if bbox is None:
                cropped = img
            else:
                x1, y1, x2, y2 = bbox
                ex1, ey1, ex2, ey2 = enlarge_bbox((x1, y1, x2, y2), w, h, ENLARGE_RATIO)
                cropped = img[ey1:ey2, ex1:ex2]

            # Save cropped frame using same filename
            cv2.imwrite(str(out_dir / frame_path.name), cropped)

# Usage Example (without sanity checks)
# if __name__ == "__main__":
#     preprocess_dataset()
