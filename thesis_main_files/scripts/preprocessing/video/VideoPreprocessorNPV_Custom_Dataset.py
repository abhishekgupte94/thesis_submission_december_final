#!/usr/bin/env python
# VideoPreprocessorNPV_TimestampDriven.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, Any

import ast
import json

import cv2
import numpy as np
import torch

# Try to reuse InsightFace if available (matches your current approach)
try:
    from insightface.app import FaceAnalysis  # type: ignore
except Exception:
    FaceAnalysis = None


@dataclass
class TimestampDrivenVideoPreprocessorConfig:
    # Keep these aligned with your existing config defaults
    detector_size: Tuple[int, int] = (640, 640)
    crop_resize: Optional[Tuple[int, int]] = (240, 240)

    insightface_model_name: str = "buffalo_l"
    ctx_id: int = 0
    use_gpu_if_available: bool = True
    providers_gpu: Tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
    providers_cpu: Tuple[str, ...] = ("CPUExecutionProvider",)

    # Output control
    keep_full_when_no_face: bool = True
    jpeg_quality: int = 95

    # Saving choices
    save_segment_mp4: bool = True                 # keep current behavior (seg_XXXX.mp4)
    save_segment_frames_pt: bool = True           # NEW: seg_XXXX_frames.pt
    pt_dtype: str = "uint8"                       # "uint8" or "float32"
    pt_layout: str = "tchw"                       # "tchw" or "thwc"


def _read_segments_list(ts_path: Union[str, Path]) -> List[Tuple[float, float]]:
    """
    Reads a timestamp file that contains something like:
        [[s1,e1],[s2,e2],...]
    Supports:
      - JSON (recommended)
      - Python literal list (via ast.literal_eval)
    Returns validated list of (start_sec, end_sec) as floats.
    """
    ts_path = Path(ts_path)
    raw = ts_path.read_text(encoding="utf-8").strip()

    data: Any
    if ts_path.suffix.lower() in {".json"}:
        data = json.loads(raw)
    else:
        # allows .txt/.list etc containing [[...], [...]]
        data = ast.literal_eval(raw)

    if not isinstance(data, list):
        raise ValueError(f"Timestamp file must contain a list, got: {type(data)}")

    segments: List[Tuple[float, float]] = []
    for i, item in enumerate(data):
        if not (isinstance(item, (list, tuple)) and len(item) >= 2):
            raise ValueError(f"Bad segment at idx={i}: expected [start,end], got: {item}")
        s = float(item[0])
        e = float(item[1])
        if e <= s:
            # skip or error — choose strict behavior to avoid silent bugs
            raise ValueError(f"Bad segment at idx={i}: end <= start ({s}, {e})")
        segments.append((s, e))

    return segments


def _write_bgr_frames_to_video(
    frames_bgr: Sequence[np.ndarray],
    out_path: Union[str, Path],
    fps: float,
    fourcc_str: str = "mp4v",
) -> Tuple[bool, int]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(frames_bgr) == 0:
        return False, 0

    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(w), int(h)))

    written = 0
    try:
        for fr in frames_bgr:
            if fr is None:
                continue
            if fr.shape[:2] != (h, w):
                fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(fr)
            written += 1
    finally:
        writer.release()

    return True, written


class TimestampDrivenVideoPreprocessorNPV:
    """
    Timestamp-driven variant of VideoPreprocessorNPV:
      - DOES NOT build word segments
      - consumes explicit [[start,end], ...] from a timestamp file
      - extracts face crops per frame in each segment
      - saves:
          <out_dir>/<video_stem>/seg_XXXX/seg_XXXX.mp4          (optional)
          <out_dir>/<video_stem>/seg_XXXX/seg_XXXX_frames.pt    (NEW, optional)
      - optionally writes a sidecar .pt containing segments + crops_root for alignment
    """

    def __init__(self, cfg: Optional[TimestampDrivenVideoPreprocessorConfig] = None):
        self.cfg = cfg or TimestampDrivenVideoPreprocessorConfig()

        self.face_app = None
        if FaceAnalysis is not None:
            try:
                providers = self.cfg.providers_gpu if self.cfg.use_gpu_if_available else self.cfg.providers_cpu

                # IMPORTANT: pin ORT CUDA provider to ctx_id (DDP-safe pattern)
                provider_options = None
                if self.cfg.use_gpu_if_available:
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
                    provider_options=provider_options,
                )
                self.face_app.prepare(ctx_id=int(self.cfg.ctx_id), det_size=self.cfg.detector_size)
            except Exception:
                self.face_app = None

    def detect_and_crop_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Matches your current semantics:
          - if InsightFace is available: use it
          - else: return None (caller decides fallback)
        """
        if self.face_app is None:
            return None

        faces = self.face_app.get(frame_bgr)
        if not faces:
            return None

        # pick the biggest face (common robust heuristic)
        best = None
        best_area = -1.0
        for f in faces:
            bbox = getattr(f, "bbox", None)
            if bbox is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

        if best is None:
            return None

        x1, y1, x2, y2 = best
        h, w = frame_bgr.shape[:2]

        # clip
        nx1 = int(max(0, min(w - 1, round(x1))))
        ny1 = int(max(0, min(h - 1, round(y1))))
        nx2 = int(max(0, min(w, round(x2))))
        ny2 = int(max(0, min(h, round(y2))))

        if nx2 <= nx1 or ny2 <= ny1:
            return None

        crop = frame_bgr[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return None

        if self.cfg.crop_resize is not None:
            crop = cv2.resize(crop, self.cfg.crop_resize, interpolation=cv2.INTER_LINEAR)

        return crop

    def _segment_frames_to_tensor(self, frames_bgr: Sequence[np.ndarray]) -> torch.Tensor:
        """
        Converts list of BGR uint8 frames into a torch tensor.
        Layout options:
          - "tchw": (T, C, H, W)
          - "thwc": (T, H, W, C)
        Dtype options:
          - "uint8": keep raw bytes (compact + faithful to pixels)
          - "float32": normalized 0..1 float
        """
        if len(frames_bgr) == 0:
            # empty tensor with sane shape
            if self.cfg.pt_layout == "thwc":
                return torch.empty((0, 0, 0, 3), dtype=torch.uint8)
            return torch.empty((0, 3, 0, 0), dtype=torch.uint8)

        arr = np.stack(frames_bgr, axis=0)  # (T, H, W, C) BGR uint8
        t = torch.from_numpy(arr)  # uint8

        if self.cfg.pt_layout == "tchw":
            t = t.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

        if self.cfg.pt_dtype == "float32":
            t = t.float().div_(255.0)

        return t

    def process_from_timestamp_file(
        self,
        video_path: Union[str, Path],
        timestamps_path: Union[str, Path],
        out_dir: Union[str, Path],
        out_sidecar_pt_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[int, int]:
        """
        Main entry:
          - reads [[s,e],...] from timestamps_path
          - extracts crops for each segment
          - saves per-segment mp4 (optional) + per-segment frames .pt (optional)
          - optional: writes a sidecar .pt with segments_sec + crops_root

        Returns:
          (num_segments, total_saved_frames_across_segments)
        """
        video_path = Path(video_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        segments_sec = _read_segments_list(timestamps_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-6 or np.isnan(fps):
            fps = 25.0

        video_root = out_dir / video_path.stem
        video_root.mkdir(parents=True, exist_ok=True)

        total_saved = 0

        try:
            for seg_idx, (seg_s, seg_e) in enumerate(segments_sec):
                start_f = int(round(float(seg_s) * float(fps)))
                end_f = int(round(float(seg_e) * float(fps)))
                if end_f <= start_f:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, start_f)))

                crops: List[np.ndarray] = []
                for _ in range(max(0, end_f - start_f)):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                    crop = self.detect_and_crop_face(frame)
                    if crop is None:
                        if self.cfg.keep_full_when_no_face:
                            crop = frame
                            if self.cfg.crop_resize is not None:
                                crop = cv2.resize(crop, self.cfg.crop_resize, interpolation=cv2.INTER_LINEAR)
                        else:
                            continue

                    crops.append(crop)

                # seg dir structure: <out>/<video_stem>/seg_XXXX/...
                seg_dir = video_root / f"seg_{seg_idx:04d}"
                seg_dir.mkdir(parents=True, exist_ok=True)

                # (A) Save mp4 per segment (your current segmentlocal behavior)
                if self.cfg.save_segment_mp4 and len(crops) > 0:
                    out_mp4 = seg_dir / f"seg_{seg_idx:04d}.mp4"
                    ok, written = _write_bgr_frames_to_video(crops, out_mp4, fps=float(fps), fourcc_str="mp4v")
                    if ok:
                        total_saved += int(written)

                # (B) NEW: Save frames tensor for that segment
                if self.cfg.save_segment_frames_pt:
                    frames_t = self._segment_frames_to_tensor(crops)
                    out_frames_pt = seg_dir / f"seg_{seg_idx:04d}_frames.pt"
                    torch.save(
                        {
                            "video_file": video_path.name,
                            "segment_index": int(seg_idx),
                            "segment_sec": (float(seg_s), float(seg_e)),
                            "fps": float(fps),
                            "frames": frames_t,  # (T,C,H,W) or (T,H,W,C)
                            "layout": self.cfg.pt_layout,
                            "dtype": self.cfg.pt_dtype,
                            "crop_resize": self.cfg.crop_resize,
                        },
                        out_frames_pt,
                    )

        finally:
            cap.release()

        # Optional alignment sidecar (similar spirit to your current .pt payload)
        if out_sidecar_pt_path is not None:
            out_sidecar_pt_path = Path(out_sidecar_pt_path)
            out_sidecar_pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "video_file": video_path.name,
                    "segments_sec": segments_sec,
                    "crops_root": str(video_root),
                    "num_segments": len(segments_sec),
                    "saved_frames": int(total_saved),
                    "timestamps_path": str(Path(timestamps_path)),
                    "config": self.cfg.__dict__,
                },
                out_sidecar_pt_path,
            )

        return len(segments_sec), int(total_saved)
