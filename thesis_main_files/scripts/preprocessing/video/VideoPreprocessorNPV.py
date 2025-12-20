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


# [ADDED] helper to convert list of BGR uint8 frames to torch uint8 (C,T,H,W)
def _crops_to_uint8_cthw(crops: List[np.ndarray]) -> torch.Tensor:
    """
    crops: List of (H, W, 3) uint8 BGR numpy arrays
    returns: (3, T, H, W) torch.uint8
    """
    # (T, H, W, 3)
    thwc = torch.from_numpy(np.stack(crops, axis=0)).to(torch.uint8)

    # (3, T, H, W)
    cthw = thwc.permute(3, 0, 1, 2).contiguous()
    return cthw


# ======================================================================
# [ADDED] Helper: write BGR frames to a single segment video (.mp4)
# - Keeps the rest of the preprocessor API unchanged.
# ======================================================================
def _write_bgr_frames_to_video(
    frames_bgr: Sequence[np.ndarray],
    out_path: Path,
    fps: float,
    fourcc_str: str = "mp4v",  # good default for .mp4 on macOS
) -> Tuple[bool, int]:
    """Write frames to video. Returns (ok, num_frames_written)."""
    if not frames_bgr:
        return False, 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    h0, w0 = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(w0), int(h0)))

    if not writer.isOpened():
        return False, 0

    written = 0
    try:
        for f in frames_bgr:
            if f is None:
                continue
            if f.shape[:2] != (h0, w0):
                f = cv2.resize(f, (w0, h0), interpolation=cv2.INTER_LINEAR)
            writer.write(f)
            written += 1
    finally:
        writer.release()

    return True, written


def _get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "thesis_main_files":
            return parent
    return here.parents[3]


def _to_rel_data_path(path: Path) -> str:
    project_root = _get_project_root()
    try:
        rel = path.resolve().relative_to(project_root)
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def load_word_times_from_csv(csv_path: Union[str, Path]) -> List[Tuple[float, float]]:
    """
    Load a CSV with at least: start,end columns (seconds).
    """
    csv_path = Path(csv_path)
    word_times: List[Tuple[float, float]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue

            # flexible column naming
            start_key = "start" if "start" in row else ("start_time" if "start_time" in row else None)
            end_key = "end" if "end" in row else ("end_time" if "end_time" in row else None)
            if start_key is None or end_key is None:
                continue

            try:
                s = float(row[start_key])
                e = float(row[end_key])
            except Exception:
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
    def __init__(self, cfg: Optional[VideoPreprocessorConfig] = None):
        self.cfg = cfg or VideoPreprocessorConfig()

        self.face_app = None
        if FaceAnalysis is not None:
            try:
                providers = self.cfg.providers_gpu if self.cfg.use_gpu_if_available else self.cfg.providers_cpu

                # ================================================================
                # [ADDED] Ensure ONNXRuntime CUDA EP uses the correct GPU per-rank.
                # Without provider_options, ORT often defaults to device_id=0,
                # causing ALL DDP ranks to run heavy evaluation_for_detection_model on GPU 0.
                #
                # We keep semantics identical:
                # - still CUDAExecutionProvider when GPU is enabled
                # - still uses ctx_id for InsightFace internal logic
                # - but now also pins ORT to ctx_id for the CUDA provider
                # ================================================================
                provider_options = None
                if self.cfg.use_gpu_if_available:
                    # Match providers order: ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    # provider_options must be same length as providers list.
                    opts = []
                    for p in list(providers):
                        if p == "CUDAExecutionProvider":
                            opts.append({"device_id": int(self.cfg.ctx_id)})
                        else:
                            opts.append({})
                    provider_options = opts

                self.face_app = FaceAnalysis(
                    name=self.cfg.insightface_model_name,
                    providers=list(providers),
                    # [ADDED] Critical to avoid all ranks using GPU 0
                    provider_options=provider_options,
                )
                self.face_app.prepare(ctx_id=int(self.cfg.ctx_id), det_size=self.cfg.detector_size)

            except Exception:
                self.face_app = None

    def build_segments_from_word_times(
        self,
        word_times: Sequence[Sequence[float]],
        target_clip_duration: float,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> List[Tuple[float, float]]:
        """
        Build segment ranges in seconds from word timestamps.
        """
        wt: List[Tuple[float, float]] = []
        for w in word_times:
            if w is None or len(w) < 2:
                continue
            try:
                s = float(w[0])
                e = float(w[1])
            except Exception:
                continue
            if e > s:
                wt.append((s, e))

        if not wt:
            return []

        wt.sort(key=lambda x: x[0])

        target = float(target_clip_duration)
        min_len = target * float(min_factor)
        max_len = target * float(max_factor)

        segments: List[Tuple[float, float]] = []
        seg_start = wt[0][0]
        seg_end = wt[0][1]

        for (ws, we) in wt[1:]:
            seg_end = max(seg_end, we)
            seg_len = seg_end - seg_start

            if seg_len >= target:
                if seg_len <= max_len:
                    segments.append((seg_start, seg_end))
                    seg_start = ws
                    seg_end = we
                else:
                    segments.append((seg_start, seg_start + max_len))
                    seg_start = ws
                    seg_end = we

        tail_len = seg_end - seg_start
        if tail_len >= min_len:
            segments.append((seg_start, seg_end))
        elif not segments:
            segments.append((seg_start, seg_end))

        return segments

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
            try:
                x1, y1, x2, y2 = f.bbox
            except Exception:
                continue
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
        """
        Returns list of segments, each segment is list of cropped face frames (BGR).
        """
        segments_sec = self.build_segments_from_word_times(
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            min_factor=min_factor,
            max_factor=max_factor,
        )
        if not segments_sec:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 25.0

        all_segment_crops: List[List[np.ndarray]] = []
        try:
            for seg_s, seg_e in segments_sec:
                start_f = int(round(float(seg_s) * float(fps)))
                end_f = int(round(float(seg_e) * float(fps)))
                if end_f <= start_f:
                    all_segment_crops.append([])
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, start_f)))

                crops: List[np.ndarray] = []
                for _ in range(max(0, end_f - start_f)):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                    crop = self.detect_and_crop_face(frame)

                    if crop is None:
                        if keep_full_when_no_face:
                            crop = frame
                        else:
                            continue

                    if self.cfg.crop_resize is not None:
                        crop = cv2.resize(crop, self.cfg.crop_resize, interpolation=cv2.INTER_LINEAR)

                    crops.append(crop)

                all_segment_crops.append(crops)
        finally:
            cap.release()

        return all_segment_crops

    def process_and_save_facecrops_to_disk_from_word_times(
            self,
            video_path: Union[str, Path],
            word_times: Sequence[Sequence[float]],
            out_dir: Union[str, Path],
            keep_full_when_no_face: bool = True,
            min_factor: float = 0.5,
            max_factor: float = 1.5,
            target_clip_duration: Optional[float] = None,
            jpeg_quality: int = 95,
            out_pt_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[int, int]:
        """
        Existing API: keep as-is.
        """
        video_path = Path(video_path)
        out_dir = Path(out_dir)

        if target_clip_duration is None:
            target_clip_duration = self.cfg.target_clip_duration_sec

        segments_sec = self.build_segments_from_word_times(
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        segment_crops = self.process_video_file_with_word_segments(
            video_path=str(video_path),
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            keep_full_when_no_face=keep_full_when_no_face,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        num_segments = len(segment_crops)
        num_words = len(word_times)

        out_dir.mkdir(parents=True, exist_ok=True)

        video_root = out_dir / video_path.stem
        video_root.mkdir(parents=True, exist_ok=True)

        total_saved = 0
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        for seg_idx, crops in enumerate(segment_crops):
            seg_dir = video_root / f"seg_{seg_idx:04d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            for j, crop_bgr in enumerate(crops):
                out_path = seg_dir / f"frame_{j:06d}{self.cfg.frame_ext}"
                ok = cv2.imwrite(str(out_path), crop_bgr, encode_params)
                if ok:
                    total_saved += 1

            # -------------------------------
            # [ADDED] Save segment as uint8 tensor in required format:
            #         "video_u8_cthw": (3, T, H, W), uint8
            # -------------------------------
            # -------------------------------
            # [PATCHED] Save segment as tensor-only uint8 (3,T,H,W)
            # -------------------------------
            if len(crops) > 0:
                video_u8_cthw = _crops_to_uint8_cthw(crops)  # (3,T,H,W) uint8
                seg_pt_path = seg_dir / f"seg_{seg_idx:04d}.pt"
                torch.save(video_u8_cthw, seg_pt_path)

        if out_pt_path is not None:
            out_pt_path = Path(out_pt_path)
            out_pt_path.parent.mkdir(parents=True, exist_ok=True)

            save_payload = {
                "video_file": video_path.name,
                "segments_sec": segments_sec,
                "crops_root": _to_rel_data_path(video_root),
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
        # ==================================================================
        # [ADDED] Optional .pt sidecar to persist segments_sec + crop root.
        # Backwards compatible: if out_pt_path is None, behavior is unchanged.
        # ==================================================================
        out_pt_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[int, int]:
        video_path = Path(video_path)
        out_dir = Path(out_dir)

        if target_clip_duration is None:
            target_clip_duration = self.cfg.target_clip_duration_sec

        # ==================================================================
        # [ADDED] Compute segments_sec here too, so the face-crop pipeline
        # can persist timestamps for downstream audio alignment.
        # Core cropping logic remains unchanged.
        # ==================================================================
        segments_sec = self.build_segments_from_word_times(
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        segment_crops = self.process_video_file_with_word_segments(
            video_path=str(video_path),
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            keep_full_when_no_face=keep_full_when_no_face,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        num_segments = len(segment_crops)
        num_words = len(word_times)

        out_dir.mkdir(parents=True, exist_ok=True)

        video_root = out_dir / video_path.stem
        video_root.mkdir(parents=True, exist_ok=True)

        total_saved = 0

        # ==============================================================
        # [CHANGED] Save ONE video per segment (instead of per-frame JPGs)
        # - Arguments/signature unchanged.
        # - total_saved remains "frames written" for backwards compatibility.
        # ==============================================================
        cap_fps = cv2.VideoCapture(str(video_path))
        if not cap_fps.isOpened():
            raise RuntimeError(f"Cannot open video for FPS read: {video_path}")
        fps = cap_fps.get(cv2.CAP_PROP_FPS)
        cap_fps.release()
        if fps is None or fps <= 0:
            fps = 25.0  # safe fallback

        for seg_idx, crops in enumerate(segment_crops):
            seg_dir = video_root / f"seg_{seg_idx:04d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            out_path = seg_dir / f"seg_{seg_idx:04d}.mp4"
            ok, written = _write_bgr_frames_to_video(crops, out_path, fps=float(fps), fourcc_str="mp4v")
            if ok:
                total_saved += int(written)

        # ==================================================================
        # [ADDED] If requested, save a .pt "alignment sidecar" that includes:
        #   - video_file
        #   - segments_sec
        #   - crops_root (directory where seg_XXXX folders are written)
        # This enables the offline trainer to drive audio slicing from video.
        # ==================================================================
        if out_pt_path is not None:
            out_pt_path = Path(out_pt_path)
            out_pt_path.parent.mkdir(parents=True, exist_ok=True)

            save_payload = {
                "video_file": video_path.name,
                "segments_sec": segments_sec,
                "crops_root": _to_rel_data_path(video_root),
                "num_segments": num_segments,
                "saved_frames": total_saved,
                "timestamps_csv": "<in_memory_word_times>",
                "pt_rel_path": _to_rel_data_path(out_pt_path),
                "config": self.cfg.__dict__,
            }
            torch.save(save_payload, out_pt_path)

        return num_segments, total_saved
