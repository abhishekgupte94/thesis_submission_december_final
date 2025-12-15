# VideoPreprocessorNPV.py (MTCNN version)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torchvision

# [NEW] MTCNN detector instead of InsightFace
try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

from PIL import Image
import time

# ----------------------------------------------------------------------
# Configuration dataclass (still optional, reused fields)
# ----------------------------------------------------------------------
@dataclass
class VideoPreprocessorConfig:
    # NOTE: These names are kept for backwards compatibility.
    # detector_name is unused for MTCNN, but kept so your code doesn't break.
    detector_name: str = "mtcnn"
    detector_ctx_id: int = 0            # GPU index if you want CUDA; 0 by default
    detector_size: Tuple[int, int] = (240, 240)  # we use detector_size[0] as MTCNN image_size
    enlarge_ratio: float = 1.3
    frame_ext: str = ".jpg"


# ----------------------------------------------------------------------
# Main Preprocessor Class
# ----------------------------------------------------------------------
class VideoPreprocessorNPV:
    """
    Video preprocessor for NPV-style pipelines.

    Uses MTCNN (facenet-pytorch) as the face detector.
    """

    def __init__(
        self,
        detector_name: str = "mtcnn",
        detector_ctx_id: int = 0,
        detector_size: Tuple[int, int] = (240, 240),
        enlarge_ratio: float = 1.3,
        frame_ext: str = ".jpg",
    ) -> None:
        self.detector_name = detector_name
        self.detector_ctx_id = detector_ctx_id
        self.detector_size = detector_size
        self.enlarge_ratio = enlarge_ratio
        self.frame_ext = frame_ext

        # MTCNN instance (created lazily)
        self._detector: Optional[MTCNN] = None

    # ------------------------------------------------------------------
    # Pickling support (for multiprocessing)
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_detector"] = None  # do not pickle the detector
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._detector = None

    # ------------------------------------------------------------------
    # Detector initialisation and helpers (MTCNN)
    # ------------------------------------------------------------------
    def _init_detector(self) -> None:
        """
        Lazily initialise the MTCNN detector (facenet-pytorch).
        """
        if self._detector is not None:
            return

        if MTCNN is None:
            raise ImportError(
                "facenet-pytorch is required for face detection but is not installed. "
                "Install via `pip install facenet-pytorch pillow`."
            )

        # Decide device: CPU on Mac by default, optional CUDA if available and ctx_id >= 0
        if torch.cuda.is_available() and self.detector_ctx_id >= 0:
            device = f"cuda:{self.detector_ctx_id}"
        else:
            device = "cpu"

        # Use the standard MTCNN cascade (not a tiny/light variant)
        # detector_size[0] used as image_size for MTCNN.
        image_size = int(self.detector_size[0])

        self._detector = MTCNN(
            image_size=image_size,
            margin=0,
            min_face_size=20,   # basic but not super tiny faces
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            keep_all=True,      # we will pick the largest below
            device=device,
            post_process=False, # we just need boxes, not aligned crops
        )

    def _detect_largest_face(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Run MTCNN on a BGR image and return the largest bbox.

        Parameters
        ----------
        img : np.ndarray
            BGR image of shape (H, W, 3), as produced by OpenCV.

        Returns
        -------
        bbox : (x1, y1, x2, y2) or None
            Pixel coordinates of the largest face, or None if no face is found.
        """
        self._init_detector()

        # Convert BGR (OpenCV) -> RGB -> PIL Image for MTCNN
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        boxes, probs = self._detector.detect(pil_img)

        if boxes is None or len(boxes) == 0:
            return None

        # Select the largest bounding box by area
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[largest_idx].astype(int)
        return x1, y1, x2, y2

    @staticmethod
    def _enlarge_bbox(
        bbox: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
        ratio: float,
    ) -> Tuple[int, int, int, int]:
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

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(img_w, nx2)
        ny2 = min(img_h, ny2)

        return nx1, ny1, nx2, ny2

    # ------------------------------------------------------------------
    # Optional in-memory video reader (torchvision)
    # ------------------------------------------------------------------
    @staticmethod
    def read_video(path: str) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
        video = video.permute(0, 3, 1, 2) / 255.0  # (T, H, W, C) -> (T, C, H, W)
        if audio.numel() > 0:
            audio = audio.permute(1, 0)            # (num_samples, channels) -> (channels, num_samples)
        return video, audio, info

    # ------------------------------------------------------------------
    # [LEGACY] Two-pass pipeline: full-frame dump + cropping from disk
    # ------------------------------------------------------------------
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
    ) -> List[Path]:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_paths: List[Path] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f"{frame_idx:04d}{self.frame_ext}"
            frame_path = output_dir_path / frame_name

            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            frame_idx += 1

        cap.release()
        print(
            f"[VideoPreprocessorNPV] Extracted {len(frame_paths)} frames "
            f"to {output_dir_path}"
        )
        return frame_paths

    def crop_faces_from_frames(
        self,
        source_frames_dir: str,
        target_frames_dir: str,
        keep_full_when_no_face: bool = True,
    ) -> List[Path]:
        src_dir = Path(source_frames_dir)
        tgt_dir = Path(target_frames_dir)
        tgt_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(
            [p for p in src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )

        cropped_paths: List[Path] = []

        if len(frame_files) == 0:
            print(f"[VideoPreprocessorNPV] No frames found in {source_frames_dir}")
            return cropped_paths

        for frame_path in tqdm(frame_files, desc="Cropping faces"):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            bbox = self._detect_largest_face(img)

            if bbox is None:
                if keep_full_when_no_face:
                    crop = img
                else:
                    continue
            else:
                ex1, ey1, ex2, ey2 = self._enlarge_bbox(
                    bbox,
                    img_w=w,
                    img_h=h,
                    ratio=self.enlarge_ratio,
                )
                crop = img[ey1:ey2, ex1:ex2]

            out_path = tgt_dir / frame_path.name
            cv2.imwrite(str(out_path), crop)
            cropped_paths.append(out_path)

        print(
            f"[VideoPreprocessorNPV] Cropped {len(cropped_paths)} frames "
            f"to {target_frames_dir}"
        )
        return cropped_paths

    def process_video_file(
        self,
        video_path: str,
        frames_root: str,
        cropped_root: str,
        video_id: Optional[str] = None,
        keep_full_when_no_face: bool = True,
    ) -> Tuple[List[Path], List[Path]]:
        video_path_str = str(video_path)
        video_stem = video_id if video_id is not None else Path(video_path_str).stem

        frames_dir = Path(frames_root) / video_stem
        cropped_dir = Path(cropped_root) / video_stem

        raw_frame_paths = self.extract_frames(video_path_str, str(frames_dir))
        cropped_paths = self.crop_faces_from_frames(
            source_frames_dir=str(frames_dir),
            target_frames_dir=str(cropped_dir),
            keep_full_when_no_face=keep_full_when_no_face,
        )
        return raw_frame_paths, cropped_paths

    # ------------------------------------------------------------------
    # NPV-style segment helpers (time -> segments -> frame indices)
    # ------------------------------------------------------------------
    def build_segments_from_word_times(
        self,
        word_times: Sequence[Sequence[float]],
        target_clip_duration: float,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> List[Tuple[float, float]]:
        if not word_times:
            return []

        segs: List[Tuple[float, float]] = []

        t_min = target_clip_duration * min_factor
        t_max = target_clip_duration * max_factor

        cur_start = float(word_times[0][0])
        cur_end = float(word_times[0][1])

        for i in range(1, len(word_times)):
            w_start = float(word_times[i][0])
            w_end = float(word_times[i][1])

            # Ensure monotonicity / small overlaps
            w_start = max(w_start, cur_end)

            tentative_end = w_end
            cur_dur = tentative_end - cur_start

            if cur_dur <= t_max:
                cur_end = tentative_end
            else:
                if (cur_end - cur_start) >= t_min:
                    segs.append((cur_start, cur_end))
                    cur_start = w_start
                    cur_end = w_end
                else:
                    segs.append((cur_start, cur_end))
                    cur_start = w_start
                    cur_end = w_end

        if (cur_end - cur_start) > 0.0:
            segs.append((cur_start, cur_end))

        return segs

    @staticmethod
    def segments_to_frame_indices(
        segments: Sequence[Tuple[float, float]],
        fps: float,
        num_frames: int,
    ) -> List[Tuple[int, int]]:
        frame_ranges: List[Tuple[int, int]] = []

        for (start_t, end_t) in segments:
            start_idx = int(round(start_t * fps))
            end_idx = int(round(end_t * fps))

            start_idx = max(0, min(start_idx, num_frames))
            end_idx = max(0, min(end_idx, num_frames))

            if end_idx <= start_idx:
                continue

            frame_ranges.append((start_idx, end_idx))

        return frame_ranges

    # ------------------------------------------------------------------
    # Word-segment-aware, single-pass video processing (in-memory crops)
    # ------------------------------------------------------------------
    def process_video_file_with_word_segments(
        self,
        video_path: str,
        cropped_root: str,
        word_times: Sequence[Sequence[float]],
        target_clip_duration: float,
        video_id: Optional[str] = None,
        keep_full_when_no_face: bool = True,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> List[List[Path]]:
        video_path_str = str(video_path)
        video_stem = video_id if video_id is not None else Path(video_path_str).stem

        video_crops_root = Path(cropped_root) / video_stem
        video_crops_root.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path_str}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 25.0

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        segments_sec = self.build_segments_from_word_times(
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        if not segments_sec:
            cap.release()
            return []

        frame_ranges = self.segments_to_frame_indices(
            segments=segments_sec,
            fps=fps,
            num_frames=num_frames,
        )

        segment_crops: List[List[Path]] = [[] for _ in frame_ranges]

        for seg_idx, (start_idx, end_idx) in enumerate(frame_ranges):
            if start_idx >= end_idx:
                continue

            seg_dir = video_crops_root / f"seg_{seg_idx:04d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            local_frame_idx = 0

            for frame_idx in range(start_idx, end_idx):
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                bbox = self._detect_largest_face(frame)

                if bbox is None:
                    if keep_full_when_no_face:
                        crop = frame
                    else:
                        continue
                else:
                    ex1, ey1, ex2, ey2 = self._enlarge_bbox(
                        bbox,
                        img_w=w,
                        img_h=h,
                        ratio=self.enlarge_ratio,
                    )
                    crop = frame[ey1:ey2, ex1:ex2]

                crop_name = f"{local_frame_idx:04d}{self.frame_ext}"
                crop_path = seg_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                segment_crops[seg_idx].append(crop_path)
                local_frame_idx += 1

        cap.release()
        return segment_crops
