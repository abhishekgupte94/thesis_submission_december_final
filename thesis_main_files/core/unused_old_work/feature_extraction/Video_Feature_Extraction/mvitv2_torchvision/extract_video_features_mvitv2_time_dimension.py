import argparse
import json
import logging
import os,tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

# Default logging (quiet); use --verbose for INFO
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# from pathlib import Path
# import tempfile, os
# import torch
# from typing import List, Optional, Dict, Any

def _atomic_torch_save(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(out_path.parent), delete=False) as tf:
        tmp = Path(tf.name)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, out_path)
    except Exception:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        raise
# ----------------- Utilities (no hardcoding) -----------------

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def infer_feature_dim_from_model(model: nn.Module) -> int:
    """
    Determine feature dimensionality from model config only:
      1) Prefer Linear.in_features in the classification head
      2) Fallback to model.norm.normalized_shape
    """
    if hasattr(model, "head"):
        last_linear: Optional[nn.Linear] = None
        for m in model.head.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None and hasattr(last_linear, "in_features"):
            return int(last_linear.in_features)

    if hasattr(model, "norm") and hasattr(model.norm, "normalized_shape"):
        ns = model.norm.normalized_shape
        if isinstance(ns, (tuple, list)) and len(ns) > 0:
            return int(ns[-1])

    raise RuntimeError("Could not infer feature_dim from model configuration")

def infer_thw_from_model(model, T_in: int, H_in: int, W_in: int):
    # 1) Patch-embed stride (conv3d)
    st = model.conv_proj.stride            # (t, h, w)
    T, H, W = T_in // st[0], H_in // st[1], W_in // st[2]

    # 2) Per-block query pooling stride (controls token count)
    for blk in model.blocks:
        pq = getattr(blk.attn, "pool_q", None)
        if pq is not None and hasattr(pq, "pool") and hasattr(pq.pool, "stride"):
            s = pq.pool.stride             # (t, h, w)
            T = max(1, T // s[0])
            H = max(1, H // s[1])
            W = max(1, W // s[2])
    return int(T), int(H), int(W)

def infer_total_stride_t_hw_from_model(model: nn.Module) -> Tuple[int, int, int]:
    """
    Infer total downsampling stride along (time, height, width) by reading
    patch embedding stride and any pooling/downsamping inside blocks.
    No dummy forward; purely from module config.
    """
    t_s = h_s = w_s = 1

    # 1) Patch embedding (often a Conv3d)
    pe = getattr(model, "patch_embed", None)
    if pe is not None:
        proj = getattr(pe, "proj", None)
        if isinstance(proj, (nn.Conv3d, nn.Conv2d)):
            s = proj.stride
            if isinstance(proj, nn.Conv3d):
                t_s *= s[0]; h_s *= s[1]; w_s *= s[2]
            else:  # Conv2d
                h_s *= s[0]; w_s *= s[1]

    # 2) Inside attention blocks: pool_q / downsample (Conv3d) may add stride
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        for blk in blocks:
            attn = getattr(blk, "attn", None)
            if attn is not None:
                pool_q = getattr(attn, "pool_q", None)
                if isinstance(pool_q, nn.Conv3d):
                    st = pool_q.stride
                    t_s *= st[0]; h_s *= st[1]; w_s *= st[2]
                # pool_kv typically does not change output grid size; ignore for grid
            downsample = getattr(blk, "downsample", None)
            if isinstance(downsample, nn.Conv3d):
                st = downsample.stride
                t_s *= st[0]; h_s *= st[1]; w_s *= st[2]

    return t_s, h_s, w_s


def grid_from_input_and_stride(T: int, H: int, W: int,
                               t_s: int, h_s: int, w_s: int) -> Tuple[int, int, int]:
    """
    Compute final grid sizes (T', H', W') from input clip (T,H,W) and total strides.
    """
    T_out = ceil_div(T, max(1, t_s))
    H_out = ceil_div(H, max(1, h_s))
    W_out = ceil_div(W, max(1, w_s))
    return T_out, H_out, W_out

def infer_thw_from_model_and_input(model: nn.Module, T_in: int, H_in: int, W_in: int) -> Tuple[int, int, int]:
    """
    Compute (T', H', W') from the actual TorchVision MViT-V2 modules:
      • conv_proj (stem) stride controls initial downsample
      • attn.pool_q.pool.stride inside blocks controls token count (Q path)
    """
    T, H, W = int(T_in), int(H_in), int(W_in)

    # --- 1) Stem conv: torchvision's MViT-V2 uses model.conv_proj (Conv3d) ---
    conv_proj = getattr(model, "conv_proj", None)
    if isinstance(conv_proj, nn.Conv3d):
        st = conv_proj.stride  # (t, h, w)
        T = max(1, (T + st[0] - 1) // st[0])
        H = max(1, (H + st[1] - 1) // st[1])
        W = max(1, (W + st[2] - 1) // st[2])

    # --- 2) Per-block query pooling: only pool_q on the Q path changes token grid ---
    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, (list, nn.ModuleList)):
        for blk in blocks:
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            pool_q = getattr(attn, "pool_q", None)

            # TorchVision uses a wrapper with .pool (Conv3d). Fallback: pool_q itself Conv3d.
            if hasattr(pool_q, "pool") and isinstance(pool_q.pool, nn.Conv3d):
                st = pool_q.pool.stride
            elif isinstance(pool_q, nn.Conv3d):
                st = pool_q.stride
            else:
                st = None

            if st is not None:
                # Only the Q path stride affects sequence length
                T = max(1, (T + st[0] - 1) // st[0])
                H = max(1, (H + st[1] - 1) // st[1])
                W = max(1, (W + st[2] - 1) // st[2])

    return int(T), int(H), int(W)


####Possible ffmpegcv workaround
# --- ffmpegcv helpers (add once) ---
from typing import List, Optional
import numpy as np

try:
    from ffmpegcv import VideoReaderNV as _VR  # GPU NVDEC
    _VR_BACKEND = "GPU (NVDEC)"
except Exception:
    try:
        from ffmpegcv import VideoReader as _VR  # CPU fallback
        _VR_BACKEND = "CPU (ffmpeg)"
    except Exception:
        _VR = None
        _VR_BACKEND = None


def _build_indices_like_cv2(total_frames: Optional[int],
                            original_fps: float,
                            sampling_interval_ms: float) -> Optional[List[int]]:
    """
    Mirror your cv2 index logic exactly (including duplicates when upsampling).
    """
    if not total_frames or total_frames <= 0:
        return None
    target_fps = 1000.0 / sampling_interval_ms  # 40ms -> 25 fps
    if abs(original_fps - target_fps) < 0.1:
        return list(range(total_frames))
    duration_ms = (total_frames / max(original_fps, 1e-6)) * 1000.0
    num_target = int(duration_ms / sampling_interval_ms)
    idxs: List[int] = []
    for i in range(num_target):
        t_ms = i * sampling_interval_ms
        idx = int((t_ms / 1000.0) * original_fps)
        if idx < total_frames:
            idxs.append(idx)  # keep duplicates exactly like cv2
    return idxs

def sample_frames_ffmpegcv_like_cv2(path,
                                    sampling_interval_ms: float = 40.0,
                                    fallback_fps: float = 25.0,
                                    resize: Optional[tuple[int,int]] = None,
                                    pix_fmt: str = "rgb24") -> List[np.ndarray]:
    """
    Decode + sample using ffmpegcv, matching cv2 semantics (RGB HWC uint8 frames).
    """
    if _VR is None:
        raise RuntimeError("ffmpegcv not available (install `ffmpegcv`).")

    frames: List[np.ndarray] = []
    with _VR(str(path), pix_fmt=pix_fmt, resize=resize) as vr:
        src_fps = float(vr.fps) if getattr(vr, "fps", None) else 0.0
        fps = src_fps if src_fps > 0 else fallback_fps
        total_frames = int(vr.count) if getattr(vr, "count", None) else None

        indices = _build_indices_like_cv2(total_frames, fps, sampling_interval_ms)

        if indices is None:
            # Count unknown -> approximate stride sampling to target fps
            target_fps = 1000.0 / sampling_interval_ms
            stride = max(1, round(fps / target_fps))
            for i, frame in enumerate(vr):
                if i % stride == 0:
                    frames.append(frame)  # (H, W, 3), RGB uint8
            return frames

        # Exact cv2-like behavior without random seeks (keep duplicates)
        it = iter(indices)
        try:
            next_idx = next(it)
        except StopIteration:
            next_idx = None

        for i, frame in enumerate(vr):
            while next_idx is not None and i == next_idx:
                frames.append(frame)  # may append same frame multiple times
                try:
                    next_idx = next(it)
                except StopIteration:
                    next_idx = None
                    break
            if next_idx is None:
                break

    return frames

######


# ----------------- Extractor -----------------
## NEW fixed GPU code
class MViTv2FeatureExtractor:
    def __init__(
            self,
            device: str = "cuda",
            preserve_temporal: bool = True,
            temporal_pool: bool = True,
            aggregate: str = "mean",
            dtype: torch.dtype = torch.float32,
            verbose: bool = False,
            default_save_dir: Optional[str] = None,
    ):
        self.preserve_temporal = preserve_temporal

        # ✅ DEVICE-ONLY FIX: Proper device initialization
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # END DEVICE FIX

        self.verbose = verbose
        self.dtype = dtype
        self.default_save_dir = Path(default_save_dir) if default_save_dir else None
        # torch.set_grad_enabled(False)

        # Pretrained model + transforms
        self.weights = MViT_V2_S_Weights.KINETICS400_V1
        self.model = mvit_v2_s(weights=self.weights).to(self.device).eval()
        self.temporal_pool = temporal_pool
        for p in self.model.parameters():
            p.requires_grad = False

        self.preprocess = self.weights.transforms()

        # Hook cache for intermediate tensors
        self.features_cache: Dict[str, torch.Tensor] = {}
        self._register_hooks()

        # Introspect config-based dims/strides (no forward)
        self.feature_dim = infer_feature_dim_from_model(self.model)
        self.total_stride_t, self.total_stride_h, self.total_stride_w = infer_total_stride_t_hw_from_model(self.model)

        logger.info(f"Inferred feature_dim: {self.feature_dim}")
        logger.info(f"Inferred total stride: t={self.total_stride_t}, h={self.total_stride_h}, w={self.total_stride_w}")

    # ... [Keep all other methods unchanged] ...
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

    # ---------- Video I/O ----------

    def _sample_frames_cv2(self,
                           cap: cv2.VideoCapture,
                           original_fps: float,
                           sampling_interval_ms: float,
                           total_frames: int) -> List[np.ndarray]:
        print(f"[ffmpegcv] Using backend: {_VR_BACKEND}")  # 👈 Added log

        frames: List[np.ndarray] = []
        target_fps = 1000.0 / sampling_interval_ms  # e.g., 40ms -> 25 fps

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
        """
        List[HWC uint8] -> (T, C, H, W) uint8
        """
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

    # ---------- Clip selection (center/uniform) ----------

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
            starts = [int(round(s)) for s in np.linspace(0, max_start, clips_per_video)] if max_start > 0 else [0]
            for s in starts:
                e = min(s + clip_len, total)
                s = max(0, e - clip_len)
                clips.append(video_tensor[s:e])

        logger.info(f"Extracted {len(clips)} clip(s) of length {clip_len} (standard selection)")
        return clips

    # ---------- Forward & helper transforms ----------

    def _forward_model(self,
                       vt_bcthw: torch.Tensor,
                       preserve_temporal: bool,
                       temporal_pool: bool,
                       input_T: int,
                       input_H: int,
                       input_W: int) -> torch.Tensor:
        """
        vt_bcthw: (B, C, T, H, W) float tensor (already preprocessed)
        Returns features on CPU:
          - if not preserve_temporal: (B, D)
          - if preserve_temporal and not temporal_pool: (B, L, D)  (CLS + patches)
          - if preserve_temporal and temporal_pool:     (B, T', D) (spatially pooled)
        """
        # GPU-friendly bits:
        self.model.eval()
        use_amp = (getattr(self, "dtype", torch.float32) == torch.float16)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            x = vt_bcthw.to(self.device, non_blocking=True)

            self.features_cache.clear()
            _ = self.model(x)

            if not preserve_temporal:
                if "pre_classifier" in self.features_cache:
                    feats = self.features_cache["pre_classifier"]  # (B, D)
                elif "norm_output" in self.features_cache:
                    tfeats = self.features_cache["norm_output"]  # (B, L, D)
                    feats = tfeats.mean(dim=1)  # pool tokens
                else:
                    feats = self.model(x)  # logits as fallback
                return feats.detach()

            # preserve_temporal == True
            if "norm_output" in self.features_cache:
                seq = self.features_cache["norm_output"]  # (B, L, D)
            elif "last_block_output" in self.features_cache:
                seq = self.features_cache["last_block_output"]
                if isinstance(seq, tuple):
                    seq = seq[0]
            else:
                # Fallback: logits with dummy token dim
                seq = self.model(x).unsqueeze(1)  # (B, 1, D)

            # Temporal pooling option: convert (B, L, D) -> (B, T', D)
            if temporal_pool and seq.dim() == 3 and seq.shape[1] >= 1:
                # Compute T',H',W' from model strides and transformed input size
                T_prime, H_prime, W_prime = infer_thw_from_model_and_input(
                    self.model, input_T, input_H, input_W
                )
                B, L, D = seq.shape
                L_no_cls = max(0, L - 1)
                if T_prime * H_prime * W_prime != L_no_cls and L_no_cls > 0:
                    # Fallback: recompute spatial grid guess from patch size 16 (typical TV MViT-V2 S)
                    h_guess = (input_H + 15) // 16
                    w_guess = (input_W + 15) // 16
                    if h_guess > 0 and w_guess > 0 and L_no_cls % (h_guess * w_guess) == 0:
                        H_prime, W_prime = h_guess, w_guess
                        T_prime = L_no_cls // (H_prime * W_prime)

                # We expect L == 1 + T'*H'*W' (CLS + patches); be robust if CLS missing
                B, L, D = seq.shape
                # Heuristic: if L == 1 + T'*H'*W', assume CLS at index 0
                cls_tokens = 1 if L == (1 + T_prime * H_prime * W_prime) else 0
                patches = seq[:, cls_tokens:, :]  # (B, T'*H'*W', D) or (B, L, D)
                if (T_prime * H_prime * W_prime) > 0 and patches.shape[1] >= (T_prime * H_prime * W_prime):
                    patches = patches[:, :T_prime * H_prime * W_prime, :]
                    patches = patches.view(B, T_prime, H_prime * W_prime, D)  # (B, T', H'*W', D)
                    feats = patches.mean(dim=2)  # (B, T', D)
                else:
                    # If shapes don't match expectation, just mean over tokens as fallback
                    logger.warning(
                        f"Temporal pool shape mismatch: L={L}, expected 1+T'H'W'={1 + T_prime * H_prime * W_prime}. "
                        f"Falling back to simple token mean per time inference.")
                    feats = seq.mean(dim=1, keepdim=False).unsqueeze(1)  # (B, 1, D)
                return feats.detach()

            # If not temporal_pool, return the raw sequence
            return seq.detach()

    def _process_clips_through_model(self,
                                     clips: List[torch.Tensor],
                                     aggregate: str,
                                     preserve_temporal: bool = False,
                                     temporal_pool: bool = False) -> np.ndarray:
        """
        For each clip (T, C, H, W) uint8:
          • Apply self.preprocess() to (T, C, H, W) tensor
          • Make (B, C, T, H, W)
          • Forward → collect features
        Aggregation depends on shapes and flags.
        """
        clip_features: List[torch.Tensor] = []

        for clip in clips:
            try:
                # Transform expects (T, C, H, W) tensor
                vt = self.preprocess(clip)
                if vt.dim() != 4:
                    raise RuntimeError(
                        f"Unexpected transform output rank: {vt.dim()}, shape={getattr(vt, 'shape', None)}")

                # Normalize to (1, C, T, H, W) regardless of what transforms() returns.
                # Case A: transforms() returned (C, T, H, W)
                if vt.shape[0] in (1, 3) and vt.shape[1] > 4:
                    C_in, T_in, H_in, W_in = int(vt.shape[0]), int(vt.shape[1]), int(vt.shape[2]), int(vt.shape[3])
                    vt = vt.unsqueeze(0)  # (1, C, T, H, W)

                # Case B: transforms() returned (T, C, H, W)
                elif vt.shape[1] in (1, 3) and vt.shape[0] > 4:
                    T_in, C_in, H_in, W_in = int(vt.shape[0]), int(vt.shape[1]), int(vt.shape[2]), int(vt.shape[3])
                    vt = vt.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

                # Fallback: be explicit and complain (helps if transforms behavior changes again)
                else:
                    raise RuntimeError(f"Ambiguous transform output shape: {tuple(vt.shape)}; "
                                       f"expected (T,C,H,W) or (C,T,H,W).")

                feats = self._forward_model(
                    vt_bcthw=vt,
                    preserve_temporal=preserve_temporal,
                    temporal_pool=temporal_pool,
                    input_T=T_in, input_H=H_in, input_W=W_in
                )
                feats_cpu = feats.squeeze(0)  # drop batch for aggregation
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"Clip features shape: {tuple(feats_cpu.shape)}")
                clip_features.append(feats_cpu)

            except Exception as e:
                logger.error(f"Error processing clip: {e}")
                # Dynamic zero fallback
                if preserve_temporal:
                    # Try to infer (T',D) if temporal_pool, else (L,D)
                    if temporal_pool:
                        T_prime, H_prime, W_prime = grid_from_input_and_stride(
                            T=T_in if 'T_in' in locals() else 16,
                            H=H_in if 'H_in' in locals() else 224,
                            W=W_in if 'W_in' in locals() else 224,
                            t_s=self.total_stride_t, h_s=self.total_stride_h, w_s=self.total_stride_w
                        )
                        clip_features.append(torch.zeros(T_prime, self.feature_dim))
                    else:
                        # If we can't be precise, produce a single-token vector to signal failure softly
                        clip_features.append(torch.zeros(1, self.feature_dim))
                else:
                    clip_features.append(torch.zeros(self.feature_dim))

        # Aggregate across clips
        if len(clip_features) == 1:
            final = clip_features[0]
        else:
            # Stack adds a clip axis
            # possible shapes per clip:
            #  - not preserve_temporal: (D,)
            #  - preserve_temporal & !temporal_pool: (L, D)
            #  - preserve_temporal & temporal_pool: (T', D)
            clip_stack = torch.stack(clip_features)  # (N_clips, ..., D)

            if not preserve_temporal:
                if aggregate == "mean":
                    final = clip_stack.mean(dim=0)         # (D,)
                elif aggregate == "max":
                    final = clip_stack.max(dim=0)[0]
                elif aggregate == "concat":
                    final = clip_stack.flatten()
                elif aggregate == "all":
                    final = clip_stack
                else:
                    final = clip_stack.mean(dim=0)

            else:
                # preserve_temporal == True
                if aggregate == "mean":
                    final = clip_stack.mean(dim=0)         # (L,D) or (T',D)
                elif aggregate == "max":
                    final = clip_stack.max(dim=0)[0]
                elif aggregate == "concat_temporal":
                    # concat along time-like axis
                    final = clip_stack.reshape(-1, clip_stack.shape[-1])  # (N_clips*L, D) or (N_clips*T', D)
                elif aggregate == "all_temporal":
                    final = clip_stack                              # (N_clips, L, D) or (N_clips, T', D)
                else:
                    final = clip_stack.mean(dim=0)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Final aggregated features shape: {tuple(final.shape)}")
        return final.numpy()
    def extract_one(
        self,
        video_path: str,
        save: bool = False,
        save_dir: Optional[str] = None,
        overwrite: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract features for a single video.

        Args:
            video_path: path to the video file
            save: if True, save the result as a .pt file
            save_dir: directory to save outputs (defaults to <video_dir>/mvit_feats_pt)
            overwrite: if False and output exists, skip

        Returns:
            result dict with:
              {
                "path": str,
                "features": torch.Tensor (CPU),
                "time_axis": torch.Tensor | None,
                "shape": tuple
              }
            or None if skipped.
        """
        p = Path(video_path)
        out_dir = Path(save_dir) if save_dir else (self.default_save_dir or p.parent / "mvit_feats_pt")
        out_path = out_dir / f"{p.stem}.pt"

        # Skip if already exists
        if save and not overwrite and out_path.exists() and out_path.stat().st_size > 0:
            if self.verbose:
                print(f"[skip] exists: {out_path}")
            return None

        # ---------------------------
        # 1) Decode & preprocess
        # ---------------------------
        # Example: decode frames with cv2/ffmpeg, preprocess using self.preprocess
        # Should produce a tensor: (1, C, T, H, W) on self.device
        # ---------------------------
        # 1) Decode & preprocess
        # ---------------------------
        p = Path(video_path)

        # cap = cv2.VideoCapture(str(p))
        # if not cap.isOpened():
        #     raise RuntimeError(f"Failed to open video: {p}")
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 0.0
        #
        # # sample frames at ~25 fps (40 ms) just like elsewhere
        # frames = self._sample_frames_cv2(
        #     cap=cap,
        #     original_fps=fps if fps > 0 else 25.0,
        #     sampling_interval_ms=40.0,
        #     total_frames=total_frames,
        # )
        # cap.release()

        # --- ffmpegcv first, cv2 fallback ---
        try:
            frames = sample_frames_ffmpegcv_like_cv2(
                path=p,  # or video_path
                sampling_interval_ms=40.0,  # ~25 fps
                fallback_fps=25.0,
                resize=None,  # or (W, H) to decode-resize
                pix_fmt="rgb24",
            )
            if len(frames) == 0:
                logger.warning(f"No frames extracted from {p} via ffmpegcv")
                return np.zeros(self.feature_dim)

        except Exception as e:
            # logger.warning(f"ffmpegcv failed for {p} ({e}); falling back to cv2")
            print("Failed to use ffmpegcv, falling back to cv2")
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {p}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 0.0

            frames = self._sample_frames_cv2(
                cap=cap,
                original_fps=fps if fps > 0 else 25.0,
                sampling_interval_ms=40.0,
                total_frames=total_frames,
            )
            cap.release()

            if len(frames) == 0:
                logger.warning(f"No frames extracted from {p} via cv2")
                return np.zeros(self.feature_dim)

        vt_tchw = self._frames_to_tensor(frames)
        # ensure we have at least clip_len frames and pick a center clip
        clip_len = 16
        vt_tchw = self._pad_short_video(vt_tchw, clip_len=clip_len)
        T_total = int(vt_tchw.shape[0])
        start = max(0, (T_total - clip_len) // 2)
        clip_tchw = vt_tchw[start:start + clip_len]  # (T, C, H, W)

        # apply your official transforms (UNCHANGED LOGIC)
        vt = self.preprocess(clip_tchw)

        # normalize to (1, C, T, H, W) for the model (same logic you use in _process_clips_through_model)
        if vt.dim() != 4:
            raise RuntimeError(f"Unexpected transform output rank: {vt.dim()}, shape={getattr(vt, 'shape', None)}")

        # Case A: transforms() returned (C, T, H, W)
        if vt.shape[0] in (1, 3) and vt.shape[1] > 4:
            C_in, T_in, H_in, W_in = int(vt.shape[0]), int(vt.shape[1]), int(vt.shape[2]), int(vt.shape[3])
            vt_bcthw = vt.unsqueeze(0)  # (1, C, T, H, W)

        # Case B: transforms() returned (T, C, H, W)
        elif vt.shape[1] in (1, 3) and vt.shape[0] > 4:
            T_in, C_in, H_in, W_in = int(vt.shape[0]), int(vt.shape[1]), int(vt.shape[2]), int(vt.shape[3])
            vt_bcthw = vt.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

        else:
            raise RuntimeError(f"Ambiguous transform output shape: {tuple(vt.shape)}; "
                               f"expected (T,C,H,W) or (C,T,H,W).")

        # ---------------------------
        # 2) Forward through model
        # ---------------------------
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.dtype==torch.float16)):
            feats = self._forward_model(
                vt_bcthw=vt_bcthw,
                preserve_temporal=self.preserve_temporal,
                temporal_pool=self.temporal_pool,
                input_T=vt_bcthw.shape[2],
                input_H=vt_bcthw.shape[3],
                input_W=vt_bcthw.shape[4],
            )

        # feats comes back on CPU from _forward_model
        feats_cpu = feats.squeeze(0)  # (D,) or (T',D)

        # Optionally create a time axis if temporal
        time_axis = None
        if feats_cpu.dim() == 2:
            time_axis = torch.arange(feats_cpu.shape[0], dtype=torch.int32)

        result = {
            "path": str(p),
            "features": feats_cpu,      # torch.Tensor (CPU)
            "time_axis": time_axis,     # torch.Tensor or None
            "shape": tuple(feats_cpu.shape),
        }

        # ---------------------------
        # 3) Save if requested
        # ---------------------------
        if save:
            payload = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}
            _atomic_torch_save(payload, out_path)

        return result

    # ---------- Public APIs ----------
    def extract_from_paths(
        self,
        paths: List[str],
        save: bool = False,
        save_dir: Optional[str] = None,
        overwrite: bool = False,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for vp in paths:
            try:
                r = self.extract_one(vp, save=save, save_dir=save_dir, overwrite=overwrite)
                if r is not None:
                    out.append(r)
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] {vp}: {e}")
        return out

    def extract_from_video_file(self,
                                video_path: str,
                                clip_len: int = 16,
                                clips_per_video: int = 1,
                                aggregate: str = "mean",
                                sampling_interval_ms: float = 40.0,
                                target_fps: float = 25.0,
                                preserve_temporal: bool = False,
                                temporal_pool: bool = False) -> np.ndarray:
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
            return self._process_clips_through_model(
                clips, aggregate,
                preserve_temporal=preserve_temporal,
                temporal_pool=temporal_pool
            )

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return np.zeros(self.feature_dim)

    def extract_from_directory(self,
                               input_dir: str,
                               output_dir: str,
                               clip_len: int = 16,
                               clips_per_video: int = 1,
                               aggregate: str = "mean",
                               sampling_interval_ms: float = 40.0,
                               preserve_temporal: bool = False,
                               temporal_pool: bool = False,
                               save_format: str = "npz",
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
            "temporal_pool": temporal_pool,
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
                preserve_temporal=preserve_temporal,
                temporal_pool=temporal_pool
            )
            if feats is not None and not np.all(feats == 0):
                features_dict[name] = feats
                metadata["video_paths"].append(str(vf))

        # Treat output_dir as a directory; save with fixed prefix
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = out_dir / "saved_video_features_without_overlap"

        if save_format in ["npz", "both"]:
            np.savez_compressed(f"{prefix}.npz", **features_dict)
        if save_format in ["pickle", "both"]:
            with open(f"{prefix}.pkl", "wb") as f:
                pickle.dump(features_dict, f)
        if include_metadata:
            with open(f"{prefix}_metadata.json", "w") as f:
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

#
# def main():
#     parser = argparse.ArgumentParser(
#         description="Extract features from video using MViTv2 with CV2 sampling and hooks"
#     )
#     parser.add_argument("--input", type=str, required=True,
#                         help="Input directory containing video or single video file")
#     parser.add_argument("--output", type=str, required=True,
#                         help="Output directory (files saved with fixed prefix)")
#     parser.add_argument("--clip_len", type=int, default=16,
#                         help="Number of frames per clip")
#     parser.add_argument("--clips_per_video", type=int, default=1,
#                         help="Number of clips per video: 1=center; >1=uniform")
#     parser.add_argument("--aggregate", type=str, default="mean",
#                         choices=["mean", "max", "concat", "all", "concat_temporal", "all_temporal"],
#                         help="How to aggregate clip features")
#     parser.add_argument("--sampling_interval_ms", type=float, default=40.0,
#                         help="Sampling interval in ms (40.0 for 25fps)")
#     parser.add_argument("--preserve_temporal", action="store_true",
#                         help="Preserve token sequences")
#     parser.add_argument("--temporal_pool", action="store_true",
#                         help="If set with --preserve_temporal, returns (B,T',D) by spatial pooling (drops CLS)")
#     parser.add_argument("--save_format", type=str, default="npz",
#                         choices=["npz", "pickle", "both"],
#                         help="Format to save features")
#     parser.add_argument("--device", type=str, default="auto",
#                         help="Device to use (mps/cuda/cpu/auto)")
#     parser.add_argument("--verbose", action="store_true",
#                         help="Enable verbose logging (INFO level)")
#
#     args = parser.parse_args()
#     logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
#
#     if args.device == "auto":
#         if torch.backends.mps.is_available():
#             device = "mps"
#         elif torch.cuda.is_available():
#             device = "cuda"
#         else:
#             device = "cpu"
#     else:
#         device = args.device
#
#     extractor = MViTv2FeatureExtractor(device=device)
#     input_path = Path(args.input)
#
#     out_dir = Path(args.output)
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     if input_path.is_file():
#         logger.info(f"Processing single video: {input_path}")
#         feats = extractor.extract_from_video_file(
#             str(input_path),
#             clip_len=args.clip_len,
#             clips_per_video=args.clips_per_video,
#             aggregate=args.aggregate,
#             sampling_interval_ms=args.sampling_interval_ms,
#             preserve_temporal=args.preserve_temporal,
#             temporal_pool=args.temporal_pool
#         )
#         vid_name = input_path.stem
#         features_dict = {vid_name: feats}
#
#         prefix = out_dir / "saved_video_features_without_overlap"
#         if args.save_format in ["npz", "both"]:
#             np.savez_compressed(f"{prefix}.npz", **features_dict)
#         if args.save_format in ["pickle", "both"]:
#             with open(f"{prefix}.pkl", "wb") as f:
#                 pickle.dump(features_dict, f)
#
#         meta = {
#             "feature_dim": extractor.feature_dim,
#             "clip_len": args.clip_len,
#             "aggregate": args.aggregate,
#             "sampling_interval_ms": args.sampling_interval_ms,
#             "preserve_temporal": args.preserve_temporal,
#             "temporal_pool": args.temporal_pool,
#             "video_paths": [str(input_path)]
#         }
#         with open(f"{prefix}_metadata.json", "w") as f:
#             json.dump(meta, f, indent=2)
#
#         logger.info(f"Features saved to: {prefix}.*")
#
#     elif input_path.is_dir():
#         logger.info(f"Processing directory: {input_path}")
#         extractor.extract_from_directory(
#             str(input_path),
#             str(out_dir),
#             clip_len=args.clip_len,
#             clips_per_video=args.clips_per_video,
#             aggregate=args.aggregate,
#             sampling_interval_ms=args.sampling_interval_ms,
#             preserve_temporal=args.preserve_temporal,
#             temporal_pool=args.temporal_pool,
#             save_format=args.save_format
#         )

#
# if __name__ == "__main__":
#     main()
