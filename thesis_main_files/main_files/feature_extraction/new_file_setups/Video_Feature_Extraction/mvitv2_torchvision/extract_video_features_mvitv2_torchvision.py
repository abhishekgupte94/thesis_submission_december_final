import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

# Default logging (quiet); --verbose raises to INFO
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MViTv2FeatureExtractor:
    """
    MViTv2 feature extractor (TorchVision):
      • CV2 sampling at ~25 fps (40 ms)
      • Standard clipping: center (1) or uniform (>1)
      • Official TV transforms (no custom mean/std)
      • Dimensions inferred from model config / runtime tensors (no magic numbers)
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load pretrained model + transforms
        self.weights = MViT_V2_S_Weights.KINETICS400_V1
        self.model = mvit_v2_s(weights=self.weights).to(self.device).eval()

        for p in self.model.parameters():
            p.requires_grad = False

        # Official preprocessing pipeline
        self.preprocess = self.weights.transforms()

        # Hook cache for intermediate tensors
        self.features_cache: Dict[str, torch.Tensor] = {}
        self._register_hooks()

        # Infer feature dimension from model (no dummy forward)
        self.feature_dim = self._infer_feature_dim()
        logger.info(f"Inferred feature_dim: {self.feature_dim}")

    # ----------------- Model Introspection -----------------

    def _register_hooks(self) -> None:
        def hook_fn(name: str):
            def hook(module, inputs, output):
                self.features_cache[name] = output
            return hook

        if hasattr(self.model, "norm"):
            self.model.norm.register_forward_hook(hook_fn("norm_output"))

        if hasattr(self.model, "blocks") and len(self.model.blocks) > 0:
            self.model.blocks[-1].register_forward_hook(hook_fn("last_block_output"))

        if hasattr(self.model, "head"):
            if hasattr(self.model.head, "dropout"):
                self.model.head.dropout.register_forward_hook(hook_fn("pre_classifier"))
            else:
                children = list(self.model.head.children())
                if len(children) >= 2:
                    children[-2].register_forward_hook(hook_fn("pre_classifier"))

    def _infer_feature_dim(self) -> int:
        # Prefer head Linear.in_features
        if hasattr(self.model, "head"):
            last_linear: Optional[nn.Linear] = None
            for m in self.model.head.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None and hasattr(last_linear, "in_features"):
                return int(last_linear.in_features)

        # Fallback: norm.normalized_shape
        if hasattr(self.model, "norm") and hasattr(self.model.norm, "normalized_shape"):
            ns = self.model.norm.normalized_shape
            if isinstance(ns, (tuple, list)) and len(ns) > 0:
                return int(ns[-1])

        raise RuntimeError("Could not infer feature_dim from model configuration")

    # ----------------- Video I/O -----------------

    def _sample_frames_cv2(self,
                           cap: cv2.VideoCapture,
                           original_fps: float,
                           sampling_interval_ms: float,
                           total_frames: int) -> List[np.ndarray]:
        """Sample frames at ~25 fps (40 ms). Return list of RGB np.uint8 frames (H, W, C)."""
        frames: List[np.ndarray] = []
        target_fps = 1000.0 / sampling_interval_ms

        if abs(original_fps - target_fps) < 0.1:
            frame_indices = list(range(total_frames))
        else:
            duration_ms = (total_frames / max(original_fps, 1e-6)) * 1000.0
            num_target = int(duration_ms / sampling_interval_ms)
            frame_indices = []
            for i in range(num_target):
                t_ms = i * sampling_interval_ms
                idx = int((t_ms / 1000.0) * original_fps)
                if idx < total_frames:
                    frame_indices.append(idx)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {idx}")
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        return frames

    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """List[HWC uint8] → (T, C, H, W) uint8 tensor."""
        if not frames:
            raise ValueError("No frames to convert")
        video_array = np.stack(frames, axis=0)         # (T, H, W, C)
        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2)  # (T, C, H, W)
        return video_tensor

    def _pad_short_video(self, video_tensor: torch.Tensor, clip_len: int) -> torch.Tensor:
        T = int(video_tensor.shape[0])
        if T < clip_len:
            pad_n = clip_len - T
            padding = torch.zeros(pad_n, *video_tensor.shape[1:], dtype=video_tensor.dtype)
            video_tensor = torch.cat([video_tensor, padding], dim=0)
            logger.warning(f"Padded video from {T} to {clip_len} frames")
        return video_tensor

    # ----------------- Clip Selection (standard) -----------------

    def _extract_clips_from_tensor(self,
                                   video_tensor: torch.Tensor,
                                   clip_len: int,
                                   clips_per_video: int = 1) -> List[torch.Tensor]:
        """
        Standard behavior:
          • 1 → center clip
          • >1 → uniform clips across [0, T-clip_len]
        """
        total = int(video_tensor.shape[0])
        clips: List[torch.Tensor] = []

        if total <= 0:
            logger.error("Empty video tensor: no frames available.")
            return clips

        if total < clip_len:
            video_tensor = self._pad_short_video(video_tensor, clip_len)
            total = clip_len

        if clips_per_video <= 1:
            start = max(0, (total - clip_len) // 2)
            end = start + clip_len
            start = max(0, min(start, total - clip_len))
            clips.append(video_tensor[start:end])
        else:
            max_start = max(0, total - clip_len)
            if clips_per_video == 1 or max_start == 0:
                starts = [0]
            else:
                starts = [int(round(s)) for s in np.linspace(0, max_start, clips_per_video)]
            for s in starts:
                e = min(s + clip_len, total)
                s = max(0, e - clip_len)
                clips.append(video_tensor[s:e])

        logger.info(f"Extracted {len(clips)} clip(s) of length {clip_len} (standard selection)")
        return clips

    # ----------------- Forward & Aggregation -----------------

    @staticmethod
    def _infer_temporal_len_from_cache_or_input(features_cache: Dict[str, torch.Tensor],
                                                vt: Optional[torch.Tensor],
                                                default_t: int) -> int:
        """
        Derive a temporal length for fallback zeros without hardcoding.
        Prefer hook outputs; else use input time (vt).
        """
        # Prefer norm_output (often [B, T', D]) or last_block_output
        cand = None
        if "norm_output" in features_cache and isinstance(features_cache["norm_output"], torch.Tensor):
            cand = features_cache["norm_output"]
        elif "last_block_output" in features_cache and isinstance(features_cache["last_block_output"], torch.Tensor):
            cand = features_cache["last_block_output"]

        if cand is not None and cand.dim() >= 3:
            return int(cand.shape[1])

        # Fall back to transform input vt (C, T, H, W)
        if vt is not None and isinstance(vt, torch.Tensor) and vt.dim() == 4:
            # try both common conventions
            if vt.shape[1] >= 1 and vt.shape[0] in (1, 3):    # (C, T, H, W)
                return int(vt.shape[1])
            if vt.shape[0] >= 1 and vt.shape[1] in (1, 3):    # (T, C, H, W)
                return int(vt.shape[0])

        # Last resort: use input T (clip length)
        return int(default_t)

    def _extract_features_from_model(self,
                                     video_tensor: torch.Tensor,
                                     preserve_temporal: bool = False) -> torch.Tensor:
        """
        Input: (1, C, T, H, W) float normalized by official transforms.
        Output: pooled or temporal features on CPU.
        """
        with torch.no_grad():
            x = video_tensor.to(self.device)
            try:
                self.features_cache.clear()
                _ = self.model(x)

                if preserve_temporal:
                    if "norm_output" in self.features_cache:
                        feats = self.features_cache["norm_output"]
                    elif "last_block_output" in self.features_cache:
                        feats = self.features_cache["last_block_output"]
                        if isinstance(feats, tuple):
                            feats = feats[0]
                    else:
                        # Fallback: logits with a dummy temporal dim
                        feats = self.model(x).unsqueeze(1)

                    if feats.dim() == 2:
                        feats = feats.unsqueeze(1)
                else:
                    if "pre_classifier" in self.features_cache:
                        feats = self.features_cache["pre_classifier"]
                    else:
                        if "norm_output" in self.features_cache:
                            tfeats = self.features_cache["norm_output"]
                            feats = tfeats.mean(dim=1)
                        else:
                            feats = self.model(x)

                return feats.detach().cpu()

            except Exception as e:
                logger.error(f"Error in feature extraction: {e}")
                raise  # Let caller decide fallback shapes

    def _process_clips_through_model(self,
                                     clips: List[torch.Tensor],
                                     aggregate: str,
                                     preserve_temporal: bool = False) -> np.ndarray:
        """
        For each clip (T, C, H, W) uint8:
          • Convert to list[PIL.Image] RGB
          • Apply self.preprocess() → usually (C, T, 224, 224) or list of (C, H, W)
          • Normalize to (1, C, T, H, W), forward, collect features
        """
        from PIL import Image
        clip_features: List[torch.Tensor] = []

        # inside _process_clips_through_model(...)
        for clip in clips:
            try:
                # clip: (T, C, H, W) uint8 on CPU — keep it as a Tensor
                # 1) Use official transforms directly on the Tensor (as per docs).
                vt = self.preprocess(clip)  # Accepts (T, C, H, W) and returns same layout with resize+center-crop+norm

                # 2) Ensure layout for model is (B, C, T, H, W)
                if vt.dim() != 4:
                    raise RuntimeError(
                        f"Unexpected transform output rank: {vt.dim()}, shape={getattr(vt, 'shape', None)}")

                # vt is (T, C, H, W) —> permute to (C, T, H, W)
                if vt.shape[1] == 3 and vt.shape[0] >= 1:
                    vt = vt.permute(1, 0, 2, 3)
                elif vt.shape[0] == 3 and vt.shape[1] >= 1:
                    # already (C, T, H, W)
                    pass
                else:
                    # Best-effort inference: put channel (3) first, time second
                    dims = list(vt.shape)
                    c_axis = next((i for i, s in enumerate(dims) if s == 3), None)
                    t_axis = 1 if c_axis == 0 else 0  # if channels at 0, time is 1; else time is 0
                    if c_axis is None:
                        raise RuntimeError(f"Cannot infer channel axis in shape {tuple(dims)}")
                    perm = [c_axis, t_axis] + [i for i in range(4) if i not in (c_axis, t_axis)]
                    vt = vt.permute(*perm)

                # 3) Add batch: (1, C, T, H, W)
                vt = vt.unsqueeze(0)

                feats = self._extract_features_from_model(vt, preserve_temporal=preserve_temporal)
                feats_cpu = feats.squeeze(0)

                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"Clip features shape: {tuple(feats_cpu.shape)}")

                clip_features.append(feats_cpu)

            except Exception as e:
                logger.error(f"Error processing clip: {e}")
                # Fallback with dynamic temporal length (no magic numbers)
                T = int(clip.shape[0])
                if preserve_temporal:
                    clip_features.append(torch.zeros(T, self.feature_dim))
                else:
                    clip_features.append(torch.zeros(self.feature_dim))

        # Aggregate across clips
        if len(clip_features) == 1:
            final = clip_features[0]
        else:
            clip_features = torch.stack(clip_features)

            if preserve_temporal:
                if aggregate == "mean":
                    final = clip_features.mean(dim=0)
                elif aggregate == "max":
                    final = clip_features.max(dim=0)[0]
                elif aggregate == "concat_temporal":
                    final = clip_features.reshape(-1, clip_features.shape[-1])
                elif aggregate == "all_temporal":
                    final = clip_features
                else:
                    final = clip_features.mean(dim=0)
            else:
                if aggregate == "mean":
                    final = clip_features.mean(dim=0)
                elif aggregate == "max":
                    final = clip_features.max(dim=0)[0]
                elif aggregate == "concat":
                    final = clip_features.flatten()
                elif aggregate == "all":
                    final = clip_features
                else:
                    final = clip_features.mean(dim=0)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Final aggregated features shape: {tuple(final.shape)}")

        return final.numpy()

    # ----------------- Public APIs -----------------

    def extract_from_video_file(self,
                                video_path: str,
                                clip_len: int = 16,
                                clips_per_video: int = 1,
                                aggregate: str = "mean",
                                sampling_interval_ms: float = 40.0,
                                target_fps: float = 25.0,
                                preserve_temporal: bool = False) -> np.ndarray:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))

            frames = self._sample_frames_cv2(cap, fps, sampling_interval_ms, total_frames)
            cap.release()

            if len(frames) == 0:
                logger.warning(f"No frames extracted from {video_path}")
                return np.zeros(self.feature_dim)

            video_tensor = self._frames_to_tensor(frames)  # (T, C, H, W) uint8
            if video_tensor.shape[0] < clip_len:
                video_tensor = self._pad_short_video(video_tensor, clip_len)

            clips = self._extract_clips_from_tensor(video_tensor, clip_len, clips_per_video)
            return self._process_clips_through_model(clips, aggregate, preserve_temporal)

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return np.zeros(self.feature_dim)

    def extract_from_directory(self,
                               input_dir: str,
                               output_file: str,
                               clip_len: int = 16,
                               clips_per_video: int = 1,
                               aggregate: str = "mean",
                               sampling_interval_ms: float = 40.0,
                               preserve_temporal: bool = False,
                               save_format: str = "npz",
                               output_prefix: Path = None,
                               include_metadata: bool = True) -> Dict:
        input_path = Path(input_dir)
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        video_files: List[Path] = []
        for ext in video_exts:
            video_files.extend(input_path.rglob(f"*{ext}"))
            video_files.extend(input_path.rglob(f"*{ext.upper()}"))

        features_dict: Dict[str, np.ndarray] = {}
        metadata = {
            "feature_dim": self.feature_dim,
            "clip_len": clip_len,
            "aggregate": aggregate,
            "sampling_interval_ms": sampling_interval_ms,
            "preserve_temporal": preserve_temporal,
            "total_videos": len(video_files),
            "video_paths": []
        }

        for vf in tqdm(video_files, desc="Extracting features"):
            name = vf.stem
            feats = self.extract_from_video_file(
                str(vf),
                clip_len=clip_len,
                clips_per_video=clips_per_video,
                aggregate=aggregate,
                sampling_interval_ms=sampling_interval_ms,
                preserve_temporal=preserve_temporal
            )
            if feats is not None and not np.all(feats == 0):
                features_dict[name] = feats
                metadata["video_paths"].append(str(vf))

        if save_format in ["npz", "both"]:
            np.savez_compressed(f"{output_prefix}.npz", **features_dict)
        if save_format in ["pickle", "both"]:
            with open(f"{output_prefix}.pkl", "wb") as f:
                pickle.dump(features_dict, f)
        if include_metadata:
            with open(f"{output_prefix}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        return {"features": features_dict, "metadata": metadata}


def load_features(feature_file: str) -> Dict:
    if feature_file.endswith(".npz"):
        data = np.load(feature_file)
        return {k: data[k] for k in data.files}
    elif feature_file.endswith(".pkl"):
        with open(feature_file, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use .npz or .pkl")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from videos using MViTv2 with CV2 sampling and hooks"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing videos or single video file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path (without extension)")
    parser.add_argument("--clip_len", type=int, default=16,
                        help="Number of frames per clip")
    parser.add_argument("--clips_per_video", type=int, default=1,
                        help="Number of clips per video: 1=center; >1=uniform")
    parser.add_argument("--aggregate", type=str, default="mean",
        choices=["mean", "max", "concat", "all", "concat_temporal", "all_temporal"],
        help="How to aggregate clip features")
    parser.add_argument("--sampling_interval_ms", type=float, default=40.0,
                        help="Sampling interval in ms (40.0 for 25fps)")
    parser.add_argument("--preserve_temporal", action="store_true",
                        help="Preserve temporal structure in features")
    parser.add_argument("--save_format", type=str, default="npz",
                        choices=["npz", "pickle", "both"],
                        help="Format to save features")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (mps/cuda/cpu/auto)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging (INFO level)")

    args = parser.parse_args()
    # In main(), after parsing args
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / "saved_video_features_without_overlap"

    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    extractor = MViTv2FeatureExtractor(device=device)
    input_path = Path(args.input)

    if input_path.is_file():
        logger.info(f"Processing single video: {input_path}")
        feats = extractor.extract_from_video_file(
            str(input_path),
            clip_len=args.clip_len,
            clips_per_video=args.clips_per_video,
            aggregate=args.aggregate,
            sampling_interval_ms=args.sampling_interval_ms,
            preserve_temporal=args.preserve_temporal
        )
        vid_name = input_path.stem
        features_dict = {vid_name: feats}
        if args.save_format in ["npz", "both"]:
            np.savez_compressed(f"{args.output}.npz", **features_dict)
        if args.save_format in ["pickle", "both"]:
            with open(f"{args.output}.pkl", "wb") as f:
                pickle.dump(features_dict, f)
        logger.info(f"Features saved to {args.output}")

    elif input_path.is_dir():
        logger.info(f"Processing directory: {input_path}")
        extractor.extract_from_directory(
            str(input_path),
            args.output,
            clip_len=args.clip_len,
            clips_per_video=args.clips_per_video,
            aggregate=args.aggregate,
            sampling_interval_ms=args.sampling_interval_ms,
            preserve_temporal=args.preserve_temporal,
            save_format=args.save_format,
            output_prefix = output_prefix
        )


if __name__ == "__main__":
    main()
