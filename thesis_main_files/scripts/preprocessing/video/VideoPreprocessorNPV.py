# video_preprocessor_npv.py

"""
VideoPreprocessorNPV

Paper-safe, NPVForensics-style video preprocessing class,
adapted for use inside PyTorch Lightning DataModules / DataLoaders.

It:
    - reads a video file
    - extracts EVERY frame (no subsampling, no clipping by default)
    - uses RetinaFace to detect the largest face per frame
    - enlarges the bounding box and crops the face region
    - saves cropped frames to disk

[DL-INTEGRATION]:
    - RetinaFace detector is lazily created per worker.
    - Detector object is excluded from pickling to avoid issues with
      num_workers>0.

[TEMPORAL CLIPPING – OPTION B]:
    - We add helpers that:
        * Build NPV-style temporal segments from word timestamps
          (same logic as in AudioPreprocessorNPV).
        * Map those segments to frame index ranges using video FPS.
        * Group per-frame face crops into clip-level lists, aligned
          with the audio segments.

    - This does NOT change the default behaviour of extract_frames()
      or process_video_file(); it simply provides additional utilities
      the Dataset can call to obtain VA-aligned frame clips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Sequence

import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch import Tensor
import torchvision

# Optional import for RetinaFace
try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None  # type: ignore


class VideoPreprocessorNPV:
    """
    VideoPreprocessorNPV

    A configurable video preprocessing class for extracting frame-level
    face crops, designed to be used in offline preprocessing *or*
    inside a Dataset/DataModule.

    Parameters
    ----------
    enlarge_ratio : float, default=1.3
        Enlargement factor for the face bounding box.

    detector_name : str, default="retinaface_r50_v1"
        Name of the RetinaFace model (insightface).

    detector_ctx_id : int, default=0
        Device id for RetinaFace. 0 = first GPU, -1 = CPU.

    detector_size : (int, int), default=(640, 640)
        Input size for the detector.

    frame_ext : str, default=".jpg"
        File extension for saved frames.

    save_visual_debug : bool, default=False
        Placeholder for future drawing/debugging hooks.
    """

    def __init__(
        self,
        enlarge_ratio: float = 1.3,
        detector_name: str = "retinaface_r50_v1",
        detector_ctx_id: int = 0,
        detector_size: Tuple[int, int] = (640, 640),
        frame_ext: str = ".jpg",
        save_visual_debug: bool = False,
    ) -> None:
        self.enlarge_ratio = enlarge_ratio
        self.detector_name = detector_name
        self.detector_ctx_id = detector_ctx_id
        self.detector_size = detector_size
        self.frame_ext = frame_ext
        self.save_visual_debug = save_visual_debug

        # [DL-INTEGRATION]: lazy detector init for DataLoader workers.
        self._detector = None  # type: ignore

        # [NEW - temporal clipping] store last seen FPS for convenience
        self._last_fps: float = 0.0

    # ------------------------------------------------------------------
    # [DL-INTEGRATION]: custom pickling to avoid detector in state
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_detector"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._detector = None

    # ------------------------------------------------------------------
    # Detector initialization
    # ------------------------------------------------------------------
    def _init_detector(self) -> None:
        """
        Lazily initialize the RetinaFace detector.

        [DL-INTEGRATION]: This is called per worker process on first use,
        to avoid non-picklable state being copied across forks.
        """
        if self._detector is not None:
            return
        if FaceAnalysis is None:
            raise ImportError(
                "insightface is required for VideoPreprocessorNPV but not installed."
            )
        self._detector = FaceAnalysis(name=self.detector_name)
        self._detector.prepare(ctx_id=self.detector_ctx_id, det_size=self.detector_size)

    # ------------------------------------------------------------------
    # Static helper to read video via torchvision (if needed in-memory)
    # ------------------------------------------------------------------
    @staticmethod
    def read_video(path: str) -> Tuple[Tensor, Tensor, dict]:
        """
        Read video and audio from file using torchvision.
        """
        video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
        video = video.permute(0, 3, 1, 2) / 255.0
        if audio.numel() > 0:
            audio = audio.permute(1, 0)
        return video, audio, info

    # ------------------------------------------------------------------
    # Face helpers
    # ------------------------------------------------------------------
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

    def _detect_largest_face(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Run RetinaFace on BGR image and return the largest face bbox.
        """
        self._init_detector()
        faces = self._detector.get(img)  # type: ignore[attr-defined]
        if len(faces) == 0:
            return None
        largest = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        x1, y1, x2, y2 = largest.bbox.astype(int)
        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Core per-video processing
    # ------------------------------------------------------------------
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
    ) -> List[Path]:
        """
        Extract EVERY frame from a video and save to disk.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_paths: List[Path] = []
        frame_idx = 1

        fps = cap.get(cv2.CAP_PROP_FPS)
        self._last_fps = float(fps) if fps > 0 else 0.0  # [NEW - temporal clipping]
        print(f"[VideoPreprocessorNPV] {video_path} | reported FPS: {fps:.2f}")

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
        """
        Detect and crop faces from EVERY frame in source_frames_dir;
        save results to target_frames_dir.
        """
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
                    bbox, img_w=w, img_h=h, ratio=self.enlarge_ratio
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
        """
        High-level pipeline:
            1) Extract every frame to frames_root/<video_id>
            2) Crop faces to cropped_root/<video_id>
        """
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

    # [DL-INTEGRATION]: optional transform-like interface for Dataset
    def __call__(self, video_path: str) -> List[Path]:
        """
        Transform-style interface for offline preprocessing inside a Dataset.

        NOTE:
            In practice you probably want to run this offline rather than
            per-sample during training, due to RetinaFace cost.
        """
        tmp_frames_root = "tmp_frames"
        tmp_cropped_root = "tmp_cropped"
        _, cropped = self.process_video_file(
            video_path=video_path,
            frames_root=tmp_frames_root,
            cropped_root=tmp_cropped_root,
            video_id=None,
            keep_full_when_no_face=True,
        )
        return cropped

    # ------------------------------------------------------------------
    # [NEW - temporal clipping] NPV-style segment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def build_segments_from_word_times(
        word_times: Sequence[Sequence[float]],
        target_clip_duration: float,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> List[Tuple[float, float]]:
        """
        Same logic as AudioPreprocessorNPV.build_segments_from_word_times,
        duplicated here so the video side can be used independently if
        desired.

        For details of behaviour and parameters, see the audio version.
        """
        if not word_times:
            return []

        word_times_sorted = sorted(word_times, key=lambda x: x[0])

        min_dur = min_factor * target_clip_duration
        max_dur = max_factor * target_clip_duration

        segments: List[Tuple[float, float]] = []
        i = 0
        n = len(word_times_sorted)

        while i < n:
            seg_start = float(word_times_sorted[i][0])
            seg_end = float(word_times_sorted[i][1])
            j = i + 1

            while j < n:
                candidate_end = float(word_times_sorted[j][1])
                candidate_dur = candidate_end - seg_start

                if candidate_dur > max_dur:
                    break

                seg_end = candidate_end
                j += 1

                if seg_end - seg_start >= target_clip_duration:
                    break

            seg_dur = seg_end - seg_start

            if seg_dur < min_dur and j < n:
                candidate_end = float(word_times_sorted[j][1])
                candidate_dur = candidate_end - seg_start
                if candidate_dur <= max_dur:
                    seg_end = candidate_end
                    j += 1
                    seg_dur = seg_end - seg_start

            if seg_end > seg_start:
                segments.append((seg_start, seg_end))

            i = max(j, i + 1)

        return segments

    @staticmethod
    def segments_to_frame_indices(
        segments: Sequence[Tuple[float, float]],
        fps: float,
        num_frames: int,
    ) -> List[Tuple[int, int]]:
        """
        Convert temporal segments (in seconds) to discrete frame index ranges
        [start_idx, end_idx) for a video with a given FPS and number of frames.
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")

        indices: List[Tuple[int, int]] = []
        for start_sec, end_sec in segments:
            start_idx = int(np.floor(start_sec * fps))
            end_idx = int(np.ceil(end_sec * fps))

            start_idx = max(0, min(start_idx, num_frames - 1))
            end_idx = max(start_idx + 1, min(end_idx, num_frames))

            indices.append((start_idx, end_idx))
        return indices

    @staticmethod
    def group_frame_paths_by_segments(
        frame_paths: Sequence[Path],
        frame_ranges: Sequence[Tuple[int, int]],
    ) -> List[List[Path]]:
        """
        Given a list of sorted frame_paths and a list of [start_idx, end_idx)
        ranges, group the paths into clip-level lists.

        Returns
        -------
        clips:
            List of lists; clips[i] contains frame paths corresponding to
            segment i. This is designed to be aligned with the audio clips
            produced by AudioPreprocessorNPV.process_file_with_word_segments().
        """
        sorted_paths = sorted(frame_paths)
        num_frames = len(sorted_paths)
        clips: List[List[Path]] = []

        for (start_idx, end_idx) in frame_ranges:
            s = max(0, min(start_idx, num_frames))
            e = max(s, min(end_idx, num_frames))
            clips.append(sorted_paths[s:e])

        return clips
