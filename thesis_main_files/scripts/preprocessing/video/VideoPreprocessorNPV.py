#!/usr/bin/env python
# VideoPreprocessorNPV.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import csv
import cv2
import numpy as np
import torch

# ======================================================================
# [ADDED] InsightFace import (FaceAnalysis uses onnxruntime / onnxruntime-gpu)
# ======================================================================
try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None  # type: ignore


def _get_project_root() -> Path:
    """Best-effort detection of the project root (``thesis_main_files``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "thesis_main_files":
            return parent
    return here.parents[3]


def _to_rel_data_path(path: Path) -> str:
    """Convert an absolute path to a POSIX-style path relative to project root."""
    project_root = _get_project_root()
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except Exception:
        return path.as_posix()


def load_word_times_from_whisper_csv(
    csv_path: Union[str, Path],
) -> List[Tuple[float, float]]:
    """Read Whisper-style word timestamps CSV -> [(start_sec, end_sec), ...]"""
    csv_path = Path(csv_path)
    word_times: List[Tuple[float, float]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = float(row["start"])
                e = float(row["end"])
            except (KeyError, ValueError):
                continue
            if e > s:
                word_times.append((s, e))

    word_times.sort(key=lambda x: x[0])
    return word_times


@dataclass
class VideoPreprocessorConfig:
    detector_size: Tuple[int, int] = (640, 640)
    crop_resize: Optional[Tuple[int, int]] = (240, 240)
    frame_ext: str = ".jpg"

    save_raw_frames: bool = False
    save_cropped_frames: bool = True
    target_clip_duration_sec: float = 2.0

    insightface_model_name: str = "buffalo_l"
    ctx_id: int = 0
    use_gpu_if_available: bool = True
    providers_gpu: Tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
    providers_cpu: Tuple[str, ...] = ("CPUExecutionProvider",)


class VideoPreprocessorNPV:
    """
    NPV-style VIDEO preprocessor (segment-based, AV-aligned).

    - Builds segments from word timestamps
    - Single-pass frame->segment assignment (seg_ptr)
    - Crops faces with InsightFace FaceAnalysis (largest face)
    """

    def __init__(self, cfg: Optional[VideoPreprocessorConfig] = None) -> None:
        self.cfg = cfg or VideoPreprocessorConfig()

        # ==================================================================
        # [ADDED] Init InsightFace FaceAnalysis once
        # ==================================================================
        self.face_app = None
        if FaceAnalysis is not None:
            providers = list(self.cfg.providers_cpu)
            ctx_id = -1
            if self.cfg.use_gpu_if_available:
                providers = list(self.cfg.providers_gpu)
                ctx_id = self.cfg.ctx_id

            self.face_app = FaceAnalysis(
                name=self.cfg.insightface_model_name,
                providers=providers,
                allowed_modules=["detection"],
            )
            self.face_app.prepare(ctx_id=ctx_id, det_size=self.cfg.detector_size)

    def build_segments_from_word_times(
        self,
        word_times: Sequence[Sequence[float]],
        target_clip_duration: float,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> List[Tuple[float, float]]:
        if not word_times:
            return []

        segments = [(float(s), float(e)) for s, e in word_times]

        def dur(seg):
            return seg[1] - seg[0]

        merged: List[Tuple[float, float]] = []
        cur = segments[0]
        for s, e in segments[1:]:
            if dur(cur) < min_factor * target_clip_duration:
                cur = (cur[0], e)
            else:
                merged.append(cur)
                cur = (s, e)
        merged.append(cur)

        final: List[Tuple[float, float]] = []
        for s, e in merged:
            d = e - s
            if d <= max_factor * target_clip_duration:
                final.append((s, e))
            else:
                n = max(int(round(d / target_clip_duration)), 1)
                step = d / n
                for i in range(n):
                    cs = s + i * step
                    ce = min(e, cs + step)
                    if ce > cs:
                        final.append((cs, ce))
        return final

    # ------------------------------------------------------------------
    # [MODIFIED] Face detection + cropping is enforced here
    # ------------------------------------------------------------------
    def detect_and_crop_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Uses InsightFace FaceAnalysis (detection-only) to crop the largest face.

        IMPORTANT:
        - InsightFace expects OpenCV images directly (BGR).
        - So we do NOT convert to RGB here.
        """
        if self.face_app is None:
            # fallback: keep pipeline alive (treat full frame as crop)
            return frame_bgr

        faces = self.face_app.get(frame_bgr)  # BGR frame directly
        if not faces:
            return None

        best_bbox = None
        best_area = -1.0
        for f in faces:
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox.astype(np.float32)
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if area > best_area:
                best_area = area
                best_bbox = (x1, y1, x2, y2)

        if best_bbox is None:
            return None

        x1, y1, x2, y2 = best_bbox
        h, w = frame_bgr.shape[:2]

        nx1 = int(max(0, np.floor(x1)))
        ny1 = int(max(0, np.floor(y1)))
        nx2 = int(min(w, np.ceil(x2)))
        ny2 = int(min(h, np.ceil(y2)))

        if nx2 <= nx1 or ny2 <= ny1:
            return None

        crop = frame_bgr[ny1:ny2, nx1:nx2]

        if self.cfg.crop_resize is not None:
            crop = cv2.resize(crop, self.cfg.crop_resize, interpolation=cv2.INTER_LINEAR)

        return crop

    def process_video_file_with_word_segments(
        self,
        video_path: str,
        word_times: Sequence[Sequence[float]],
        target_clip_duration: float,
        keep_full_when_no_face: bool = True,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> List[List[np.ndarray]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        segments_sec = self.build_segments_from_word_times(
            word_times,
            target_clip_duration,
            min_factor,
            max_factor,
        )

        if not segments_sec:
            cap.release()
            return []

        def t2f(t: float) -> int:
            return int(round(t * fps))

        segments_fidx: List[Tuple[int, int]] = []
        for s, e in segments_sec:
            fs = max(t2f(float(s)), 0)
            fe = min(t2f(float(e)), num_frames)
            if fe > fs:
                segments_fidx.append((fs, fe))

        if not segments_fidx:
            cap.release()
            return []

        segment_crops: List[List[np.ndarray]] = [[] for _ in segments_fidx]

        seg_ptr = 0
        frame_idx = 0
        ok, frame = cap.read()

        while ok and frame_idx < num_frames and seg_ptr < len(segments_fidx):

            while seg_ptr < len(segments_fidx) and frame_idx >= segments_fidx[seg_ptr][1]:
                seg_ptr += 1

            if seg_ptr >= len(segments_fidx):
                break

            fs, fe = segments_fidx[seg_ptr]

            if fs <= frame_idx < fe:
                crop = self.detect_and_crop_face(frame)
                if crop is None and keep_full_when_no_face:
                    crop = frame
                if crop is not None:
                    segment_crops[seg_ptr].append(crop)

            frame_idx += 1
            ok, frame = cap.read()

        cap.release()
        return segment_crops

    def process_video_file_with_word_segments_tensor(
        self,
        video_path: str,
        word_times: Sequence[Sequence[float]],
        target_clip_duration: float,
        keep_full_when_no_face: bool = True,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> List[torch.Tensor]:
        segment_crops = self.process_video_file_with_word_segments(
            video_path,
            word_times,
            target_clip_duration,
            keep_full_when_no_face,
            min_factor,
            max_factor,
        )

        segment_tensors: List[torch.Tensor] = []
        for crops in segment_crops:
            if not crops:
                segment_tensors.append(torch.empty(0, 3, 0, 0))
                continue

            frames = []
            for f in crops:
                f = f.astype("float32") / 255.0
                frames.append(np.transpose(f, (2, 0, 1)))
            arr = np.stack(frames, axis=0)
            segment_tensors.append(torch.from_numpy(arr))

        return segment_tensors

    def process_and_save_from_timestamps_csv_segmentlocal(
        self,
        video_path: Union[str, Path],
        word_times: Sequence[Sequence[float]],
        out_pt_path: Union[str, Path],
        keep_full_when_no_face: bool = True,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
        target_clip_duration: Optional[float] = None,
    ) -> Tuple[int, int]:
        video_path = Path(video_path)
        out_pt_path = Path(out_pt_path)

        # ==================================================================
        # [MODIFIED] Accept timestamps in-memory (list of [start_sec, end_sec])
        # NOTE: Segment extraction still calls self.detect_and_crop_face(...)
        # ==================================================================
        num_words = len(word_times)

        if target_clip_duration is None:
            target_clip_duration = self.cfg.target_clip_duration_sec

        segment_tensors = self.process_video_file_with_word_segments_tensor(
            video_path=str(video_path),
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            keep_full_when_no_face=keep_full_when_no_face,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        num_segments = len(segment_tensors)
        out_pt_path.parent.mkdir(parents=True, exist_ok=True)

        save_payload = {
            "video_file": video_path.name,
            "video_segments": segment_tensors,
            "num_segments": num_segments,
            "num_words": num_words,
            "timestamps_csv": "<in_memory_word_times>",
            "pt_rel_path": _to_rel_data_path(out_pt_path),
            "config": self.cfg.__dict__,
        }
        torch.save(save_payload, out_pt_path)

        return num_segments, num_words

    def process_and_save_facecrops_to_disk_from_word_times_segmentlocal(
        self,
        video_path: Union[str, Path],
        word_times: Sequence[Sequence[float]],
        out_dir: Union[str, Path],
        keep_full_when_no_face: bool = True,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
        target_clip_duration: Optional[float] = None,
        jpeg_quality: int = 95,
    ) -> Tuple[int, int]:
        video_path = Path(video_path)
        out_dir = Path(out_dir)

        if target_clip_duration is None:
            target_clip_duration = self.cfg.target_clip_duration_sec

        segment_crops = self.process_video_file_with_word_segments(
            video_path=str(video_path),
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            keep_full_when_no_face=keep_full_when_no_face,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        num_segments = len(segment_crops)
        total_saved = 0

        video_root = out_dir / video_path.stem
        video_root.mkdir(parents=True, exist_ok=True)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        for seg_idx, crops in enumerate(segment_crops):
            seg_dir = video_root / f"seg_{seg_idx:04d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            for j, crop_bgr in enumerate(crops):
                out_path = seg_dir / f"frame_{j:06d}{self.cfg.frame_ext}"
                ok = cv2.imwrite(str(out_path), crop_bgr, encode_params)
                if ok:
                    total_saved += 1

        return num_segments, total_saved
