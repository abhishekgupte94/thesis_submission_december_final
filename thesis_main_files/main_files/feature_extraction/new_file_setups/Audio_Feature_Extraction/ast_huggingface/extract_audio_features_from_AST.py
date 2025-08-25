# ast_level1_extractor_v2_dir_astmod
# ast_level1_extractor_v2_dir_astmodel.py
# Directory-based AST feature extractor (no CSV).
# - Supports AUDIO files or VIDEO files (demux audio via ffmpeg).
# - Uses ASTModel (backbone only).
# - Extracts Level-1 features: post-MHA residual, pre-MLP.
# - Optional: reshape to time-series (avg over freq patches) and/or pool.
# - Enhanced with automatic grid inference and time axis generation.
#
# Example:
#   python ast_level1_extractor_v2_dir_astmodel.py \
#     --data-dir /path/to/videos_or_audio \
#     --source-type video \
#     --exts .mp4,.mov \
#     --recursive yes \
#     --sr 16000 --num-mel-bins 128 --max-length 1024 \
#     --batch-size 8 --time-series yes --token-pool none \
#     --return-time-axis yes --center-time yes \
#     --out output_features

import os
import io
import json
import argparse
import subprocess
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import Resample

from dataclasses import dataclass
from transformers import ASTFeatureExtractor, ASTModel, ASTConfig


# =====================
# Enhanced Time-Series Functions
# =====================

def _cfg_get(cfg, *names, default=None):
    """Helper to read config safely with multiple name variants"""
    for n in names:
        if hasattr(cfg, n):
            v = getattr(cfg, n)
            if v is not None:
                return v
        if isinstance(cfg, dict) and n in cfg and cfg[n] is not None:
            return cfg[n]
    return default

def count_specials(embeddings_module) -> int:
    """Count special tokens (CLS, distillation) in embeddings module"""
    n = 0
    if hasattr(embeddings_module, "cls_token") and embeddings_module.cls_token is not None:
        n += 1
    if hasattr(embeddings_module, "distillation_token") and embeddings_module.distillation_token is not None:
        n += 1
    if hasattr(embeddings_module, "distill_token") and embeddings_module.distill_token is not None:
        n += 1
    return n

def ast_grid_from_config(
    model,
    feat_extractor,
    L_incl_specials: Optional[int] = None,
) -> Dict[str, float]:
    """
    Returns a dict with nf, nt, pf, pt, stride_f, stride_t, specials, N (patch tokens).
    Prefers config-declared strides; otherwise infers grid from token count and geometry.
    """
    cfg = model.config
    # patch size
    patch_size = _cfg_get(cfg, "patch_size", default=16)
    if isinstance(patch_size, int):
        pf = pt = int(patch_size)
    else:
        pf, pt = int(patch_size[0]), int(patch_size[1])

    # spectrogram geometry (from FE)
    F = getattr(feat_extractor, "num_mel_bins", 128)
    T = getattr(feat_extractor, "max_length", 1024)

    # try to read strides from config
    stride_f = _cfg_get(cfg, "fstride", "freq_stride", "stride_f", default=None)
    stride_t = _cfg_get(cfg, "tstride", "time_stride", "stride_t", default=None)
    # HF sometimes stores one 'stride' for both axes
    if stride_f is None and stride_t is None:
        s = _cfg_get(cfg, "stride", default=None)
        if s is not None:
            if isinstance(s, (tuple, list)) and len(s) == 2:
                stride_f, stride_t = int(s[0]), int(s[1])
            elif isinstance(s, int):
                stride_f = stride_t = int(s)

    # count specials if possible
    specials = count_specials(model.embeddings)

    # If both strides are available, compute nf/nt directly (non-guessy)
    if stride_f is not None and stride_t is not None:
        nf = int(math.floor((F - pf) / float(stride_f)) + 1)
        nt = int(math.floor((T - pt) / float(stride_t)) + 1)
        N = nf * nt
        return {
            "nf": nf, "nt": nt,
            "pf": pf, "pt": pt,
            "stride_f": float(stride_f), "stride_t": float(stride_t),
            "specials": specials, "N": N,
            "F": F, "T": T,
        }

    # Otherwise: infer from the *observed* token count if provided
    if L_incl_specials is None:
        raise ValueError(
            "Stride not present in config; please pass L_incl_specials (embedded sequence length) "
            "so we can infer (nf, nt) from tokens."
        )
    N = int(L_incl_specials) - int(specials)  # patch tokens only

    # Choose the (nf, nt) factor pair consistent with geometry (and reasonable strides)
    def divisors(n: int):
        out = []
        r = int(math.sqrt(n))
        for d in range(1, r + 1):
            if n % d == 0:
                out.append(d)
                if d * d != n:
                    out.append(n // d)
        return sorted(out)

    best = None
    for nf in divisors(N):
        nt = N // nf
        # implied strides (frames) if we hit exactly nf, nt
        sf = (F - pf) / (nf - 1) if nf > 1 else float("inf")
        st = (T - pt) / (nt - 1) if nt > 1 else float("inf")

        # valid if stride <= patch and positive
        def ok(x): return x > 0 and x <= max(pf, pt) + 1e-6

        if ok(sf) and ok(st):
            # favor near-integer strides, and time stride not crazy different from freq stride
            int_err = abs(sf - round(sf)) + abs(st - round(st))
            sym_err = abs(sf - st)
            score = int_err + 0.25 * sym_err
            cand = (score, nf, nt, sf, st)
            best = cand if (best is None or score < best[0]) else best

    if best is None:
        raise RuntimeError(f"Could not infer a plausible grid for N={N}, F={F}, T={T}, pf={pf}, pt={pt}.")

    _, nf, nt, sf, st = best
    return {
        "nf": int(nf), "nt": int(nt),
        "pf": pf, "pt": pt,
        "stride_f": float(sf), "stride_t": float(st),
        "specials": specials, "N": int(N),
        "F": F, "T": T,
    }

def tokens_to_time_series_auto(
    tokens: torch.Tensor,          # (B, N, C) -> patch tokens only (specials removed)
    model,                         # ASTModel
    feat_extractor,                # ASTFeatureExtractor
    *,
    L_incl_specials: Optional[int] = None,  # pass tok_all.size(1) if config lacks stride
    sample_rate: int = 16000,
    hop_length: int = 160,
    return_time_axis: bool = False,
    center: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    """
    Version of tokens_to_time_series that derives (nf, nt) and stride from the model config.
    If the config doesn't expose stride, pass L_incl_specials and it will infer grid from tokens.
    Returns: (ts, time_axis or None, info_dict)
    """
    info = ast_grid_from_config(model, feat_extractor, L_incl_specials=L_incl_specials)
    nf, nt, pt = info["nf"], info["nt"], info["pt"]

    B, N, C = tokens.shape
    if nf * nt != N:
        raise ValueError(f"N={N} tokens does not match nf*nt={nf*nt}. Did you drop specials correctly?")

    grid = tokens.view(B, nf, nt, C)
    ts = grid.mean(dim=1)  # (B, nt, C)

    if not return_time_axis:
        return ts, None, info

    # stride in frames (use config if present; else inferred in info)
    stride_frames = info["stride_t"]
    time_per_patch = (stride_frames * hop_length) / float(sample_rate)

    offset = 0.0
    if center:
        patch_duration = (pt * hop_length) / float(sample_rate)
        offset = 0.5 * patch_duration

    t = torch.arange(nt, device=tokens.device, dtype=tokens.dtype) * time_per_patch + offset
    return ts, t, info


# =====================
# Dataset: directory IO
# =====================
class DirAudioDataset(Dataset):
    """
    Scans a directory for files with given extensions and returns audio tensors.
    If source_type='video', uses ffmpeg to demux audio to WAV in-memory.
    If source_type='audio', loads with torchaudio.load.
    """
    def __init__(
        self,
        data_dir: str,
        exts: List[str],
        source_type: str = "video",   # "video" | "audio"
        target_sr: int = 16000,
        mono: bool = True,
        recursive: bool = True,
        verbose: bool = False,
    ):
        assert source_type in {"video", "audio"}
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data dir not found: {self.data_dir}")
        self.exts = {e.lower() for e in exts}
        self.source_type = source_type
        self.target_sr = target_sr
        self.mono = mono
        self.verbose = verbose

        self.paths = self._scan(self.data_dir, recursive)
        if not self.paths:
            raise FileNotFoundError(f"No files with extensions {sorted(self.exts)} in {self.data_dir}")
        if verbose:
            print(f"[DirAudioDataset] Found {len(self.paths)} files under {self.data_dir}")

    def _scan(self, root: Path, recursive: bool) -> List[str]:
        it = root.rglob("*") if recursive else root.glob("*")
        files = [str(p.resolve()) for p in it if p.is_file() and p.suffix.lower() in self.exts]
        files.sort()
        return files

    # --- video→audio demux via ffmpeg to WAV in memory ---
    def _extract_waveform_from_video(self, video_path: str) -> torch.Tensor:
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", video_path,
            "-f", "wav", "-acodec", "pcm_s16le",
            "-ar", str(self.target_sr),
            "-ac", "1", "pipe:1"
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg error for '{video_path}': {p.stderr.decode()}")
        wav_bytes = io.BytesIO(p.stdout)
        wav, sr = torchaudio.load(wav_bytes)  # (C, N)
        if self.mono and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = Resample(orig_freq=sr, new_freq=self.target_sr)(wav)
        return wav.squeeze(0)  # (N,)

    # --- direct audio load ---
    def _load_audio_file(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if self.mono and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = Resample(orig_freq=sr, new_freq=self.target_sr)(wav)
        return wav.squeeze(0)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.paths[idx]
        if self.source_type == "video":
            audio = self._extract_waveform_from_video(path)
        else:
            audio = self._load_audio_file(path)
        return {"path": path, "audio": audio, "sampling_rate": self.target_sr}


# =========================
# Picklable AST collator
# =========================
@dataclass
class ASTCollator:
    fe: ASTFeatureExtractor
    target_sr: int

    def __call__(self, batch: List[Dict]):
        arrays = [b["audio"].numpy() for b in batch]
        srs = [b["sampling_rate"] for b in batch]
        if any(sr != self.target_sr for sr in srs):
            raise RuntimeError("Sampling rate mismatch in batch after resampling.")
        out = self.fe(arrays, sampling_rate=self.target_sr, return_tensors="pt")
        return {
            "input_values": out["input_values"],  # (B, 1, n_mels, T)
            "paths": [b["path"] for b in batch],
        }


# ===========================
# Level-1 (post-MHA) features
# ===========================
def ast_embed(backbone: ASTModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Safe embedding step across versions: prefers backbone.embeddings(pixel_values),
    otherwise falls back to patch_embed + pos/cls if present.
    """
    if hasattr(backbone, "embeddings"):
        return backbone.embeddings(pixel_values)
    if hasattr(backbone, "patch_embed"):
        x = backbone.patch_embed(pixel_values)
        if hasattr(backbone, "_pos_embed"):
            x = backbone._pos_embed(x)
        if hasattr(backbone, "pos_drop"):
            x = backbone.pos_drop(x)
        return x
    raise AttributeError("AST backbone missing 'embeddings' and 'patch_embed'")

@torch.no_grad()
def ast_level1_post_mha_tokens(ast_backbone: ASTModel, pixel_values: torch.Tensor, remove_cls: bool = True) -> torch.Tensor:
    """
    Capture y = x + Attn(LN(x)) BEFORE the MLP on the FINAL encoder block.
    Returns tokens without CLS: (B, N_tokens, C)
    """
    # same embedding call as before
    x = ast_embed(ast_backbone, pixel_values)  # (B, 1+N_tokens, C)

    layers = getattr(ast_backbone.encoder, "layer",
             getattr(ast_backbone.encoder, "layers", None))
    if layers is None:
        raise AttributeError("AST backbone has no encoder layers collection")

    for block in layers:
        x_ln1 = block.layernorm_before(x)
        attn_out = block.attention(x_ln1)
        if isinstance(attn_out, tuple):  # if attention returns (output, weights)
            attn_out = attn_out[0]
        y = x + attn_out  # post-MHA residual  <-- capture point

        # advance to next block input
        # AFTER (correct for ASTLayer with Intermediate/Output)
        y_ln2 = block.layernorm_after(y)
        intermediate_out = block.intermediate(y_ln2)
        x = block.output(intermediate_out, y)  # output() handles the residual internally

        last_post_mha = y

    tokens = last_post_mha
    if remove_cls:
        tokens = tokens[:, 1:, :]
    return tokens  # (B, N_tokens, C)


def pool_tokens(tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    Pool (B, N_tokens, C) or (B, T_patches, C) → (B, C)
    """
    if mode == "mean":
        return tokens.mean(dim=1)
    if mode == "max":
        return tokens.max(dim=1).values
    raise ValueError("pool mode must be 'mean' or 'max'")


# =====
# Main
# =====
# def main():
#     ap = argparse.ArgumentParser()
#     # data (no CSV)
#     ap.add_argument("--data-dir", required=True, help="Directory containing media files")
#     ap.add_argument("--exts", default=".mp4,.mov,.mkv,.avi,.wav,.flac,.mp3",
#                     help="Comma-separated extensions to include (lowercase)")
#     ap.add_argument("--source-type", choices=["video", "audio"], default="video")
#     ap.add_argument("--recursive", choices=["yes", "no"], default="no")
#     # AST / extraction
#     ap.add_argument("--model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
#     ap.add_argument("--sr", type=int, default=16000)
#     ap.add_argument("--num-mel-bins", type=int, default=128)
#     ap.add_argument("--max-length", type=int, default=1024)
#     ap.add_argument("--batch-size", type=int, default=8)
#     ap.add_argument("--num-workers", type=int, default=0)  # 0 avoids pickling issues on macOS; raise if desired
#     ap.add_argument("--fp16", action="store_true")
#     # representation shaping
#     ap.add_argument("--time-series", choices=["yes", "no"], default="no",
#                     help="If 'yes', collapse freq patches to (B, T_patches, C).")
#     ap.add_argument("--token-pool", choices=["none", "mean", "max"], default="mean",
#                     help="Pooling over tokens/time. 'none' keeps sequence.")
#     # Enhanced debug and processing options
#     ap.add_argument("--debug-tokens", action="store_true",
#                     help="Print detailed token information for debugging")
#     ap.add_argument("--return-time-axis", choices=["yes", "no"], default="no",
#                     help="If 'yes', return time axis array with time-series features.")
#     ap.add_argument("--center-time", choices=["yes", "no"], default="no",
#                     help="If 'yes', center time axis at patch centers.")
#     ap.add_argument("--hop-length", type=int, default=160,
#                     help="STFT hop length for time axis calculation (default: 160)")
#     # output
#     ap.add_argument("--out", required=True, help="Output directory path (e.g., 'output_features' or 'output_features/')")
#     ap.add_argument("--verbose", action="store_true")
#     args = ap.parse_args()
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if args.verbose:
#         print(f"Device: {device}")
#
#     exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
#     recursive = (args.recursive.lower() == "yes")
#     return_time_axis = (args.return_time_axis.lower() == "yes")
#     center_time = (args.center_time.lower() == "yes")
#     debug_tokens = args.debug_tokens
#
#     # Dataset
#     ds = DirAudioDataset(
#         data_dir=args.data_dir,
#         exts=exts,
#         source_type=args.source_type,
#         target_sr=args.sr,
#         recursive=recursive,
#         verbose=args.verbose,
#     )
#
#     # Feature extractor (AST front-end)
#     fe = ASTFeatureExtractor(
#         sampling_rate=args.sr,
#         num_mel_bins=args.num_mel_bins,
#         max_length=args.max_length,
#         do_normalize=True,
#     )
#
#     collate = ASTCollator(fe=fe, target_sr=args.sr)
#     dl = DataLoader(
#         ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
#         pin_memory=(device.type == "cuda"), collate_fn=collate, drop_last=False
#     )
#
#     # Load model with config
#     cfg = ASTConfig.from_pretrained(args.model)
#     cfg.num_mel_bins = args.num_mel_bins
#     cfg.max_length = args.max_length
#     # Set stride if not present (for non-overlapping patches)
#     if not hasattr(cfg, "stride") and not hasattr(cfg, "freq_stride"):
#         cfg.stride = 16  # Non-overlapping stride
#
#     model = ASTModel.from_pretrained(
#         args.model,
#         config=cfg,
#         ignore_mismatched_sizes=True
#     ).to(device).eval()
#
#     # Extract
#     feats_list, paths_list = [], []
#     time_axes_list = []  # Store time axes if requested
#     grid_info = None  # Store grid information
#
#     with torch.no_grad():
#         for batch in tqdm(dl, desc="Extracting (Level 1: post-MHA)", ncols=100):
#             x = batch["input_values"].to(device)  # (B, 1, n_mels, T)
#
#             if args.verbose or debug_tokens:
#                 print("input to embed:", x.shape)
#                 print(f"Model config patch size: {getattr(model.config, 'patch_size', 'Not found')}")
#                 print(f"Feature extractor - mel bins: {fe.num_mel_bins}, max length: {fe.max_length}")
#
#             # Get tokens with special tokens for grid inference
#             tokens_with_specials = ast_level1_post_mha_tokens(model, x, remove_cls=False)  # (B, total_seq_len, 768)
#
#             if args.verbose or debug_tokens:
#                 print(f"Tokens with specials shape: {tokens_with_specials.shape}")
#
#             # Count and remove special tokens more carefully
#             specials_count = count_specials(model.embeddings)
#             if args.verbose or debug_tokens:
#                 print(f"Detected {specials_count} special tokens")
#                 print(f"CLS token present: {hasattr(model.embeddings, 'cls_token')}")
#                 if hasattr(model.embeddings, 'cls_token'):
#                     print(
#                         f"CLS token shape: {model.embeddings.cls_token.shape if model.embeddings.cls_token is not None else 'None'}")
#
#             # Remove special tokens (typically CLS token at position 0)
#             tokens = tokens_with_specials[:, specials_count:, :]  # Remove specials from beginning
#
#             if args.verbose or debug_tokens:
#                 print(f"Tokens after removing specials: {tokens.shape}")
#                 print(f"Expected grid calculation: F={fe.num_mel_bins}, T={fe.max_length}")
#                 patch_size = getattr(model.config, 'patch_size', 16)
#                 if isinstance(patch_size, int):
#                     pf = pt = patch_size
#                 else:
#                     pf, pt = patch_size[0], patch_size[1]
#                 print(f"Patch size: {pf}x{pt}")
#                 expected_nf = fe.num_mel_bins // pf
#                 expected_nt = fe.max_length // pt
#                 expected_patches = expected_nf * expected_nt
#                 print(f"Expected patches: {expected_nf}×{expected_nt} = {expected_patches}")
#                 print(f"Actual patch tokens: {tokens.shape[1]}")
#
#             if args.time_series == "yes":
#                 # Use enhanced time-series conversion
#                 try:
#                     ts, time_axis, info = tokens_to_time_series_auto(
#                         tokens,
#                         model,
#                         fe,
#                         L_incl_specials=tokens_with_specials.shape[1],  # Pass full sequence length
#                         sample_rate=args.sr,
#                         hop_length=args.hop_length,
#                         return_time_axis=return_time_axis,
#                         center=center_time
#                     )
#                     if debug_tokens:
#                         print(f"Grid info from auto function: {info}")
#
#                 except ValueError as e:
#                     if args.verbose or debug_tokens:
#                         print(f"Time-series conversion error: {e}")
#                         print(f"Attempting fallback with different special token handling...")
#
#                     # Fallback: try different special token removal
#                     if tokens_with_specials.shape[1] > tokens.shape[1] + 1:
#                         # Multiple special tokens, try removing just 1 (CLS)
#                         tokens_fallback = tokens_with_specials[:, 1:, :]
#                     else:
#                         tokens_fallback = tokens
#
#                     if debug_tokens:
#                         print(f"Fallback tokens shape: {tokens_fallback.shape}")
#
#                     ts, time_axis, info = tokens_to_time_series_auto(
#                         tokens_fallback,
#                         model,
#                         fe,
#                         L_incl_specials=tokens_with_specials.shape[1],
#                         sample_rate=args.sr,
#                         hop_length=args.hop_length,
#                         return_time_axis=return_time_axis,
#                         center=center_time
#                     )
#                     tokens = tokens_fallback
#                 rep = ts  # (B, nt, C)
#
#                 # Store grid info (same for all batches)
#                 if grid_info is None:
#                     grid_info = info
#
#                 # Store time axis if returned
#                 if time_axis is not None:
#                     time_axes_list.append(time_axis.cpu().numpy())
#             else:
#                 rep = tokens  # (B, N_tokens, C)
#
#             if args.token_pool != "none":
#                 rep = pool_tokens(rep, mode=args.token_pool)  # (B, C)
#
#             feats_list.append(rep.cpu().float().numpy())
#             paths_list.extend(batch["paths"])
#
#     feats = np.concatenate(feats_list, axis=0)
#     payload = {"features": feats, "paths": np.array(paths_list)}
#
#     # Add time axis to payload if available
#     if time_axes_list:
#         # All time axes should be identical, so just take the first one
#         payload["time_axis"] = time_axes_list[0]
#
#     meta = {
#         "data_dir": str(Path(args.data_dir).resolve()),
#         "exts": exts,
#         "recursive": recursive,
#         "source_type": args.source_type,
#         "model": args.model,
#         "sr": args.sr,
#         "num_mel_bins": args.num_mel_bins,
#         "max_length": args.max_length,
#         "hop_length": args.hop_length,
#         "hidden_size": int(model.config.hidden_size),
#         "time_series": args.time_series,
#         "token_pool": args.token_pool,
#         "return_time_axis": return_time_axis,
#         "center_time": center_time,
#         "fp16": bool(args.fp16 and device.type == "cuda"),
#         "feature_level": "Level-1 post-MHA residual (pre-MLP)",
#         "output_shape": list(feats.shape),
#     }
#
#     # Add grid information to metadata if available
#     if grid_info is not None:
#         meta["grid_info"] = {
#             "nf": grid_info["nf"],  # Number of frequency patches
#             "nt": grid_info["nt"],  # Number of time patches
#             "pf": grid_info["pf"],  # Frequency patch size
#             "pt": grid_info["pt"],  # Time patch size
#             "stride_f": grid_info["stride_f"],  # Frequency stride
#             "stride_t": grid_info["stride_t"],  # Time stride
#             "specials": grid_info["specials"],  # Number of special tokens
#             "N": grid_info["N"],  # Total patch tokens
#             "F": grid_info["F"],  # Input frequency bins
#             "T": grid_info["T"],  # Input time frames
#         }
#
#     # Your working code is now in the script:
#     # --- Save to <OUT_DIR>/audio_set.{npz,meta.json} ---
#     out_dir = Path(args.out).expanduser().resolve()  # treat --out as a directory
#     out_dir.mkdir(parents=True, exist_ok=True)
#     base = out_dir / "audio_set"
#
#     print(f"Output directory: {out_dir}")
#     print(f"Saving: {base.name}.npz and {base.name}.meta.json")
#
#     np.savez_compressed(str(base) + ".npz", **payload)  # → <OUT_DIR>/audio_set.npz
#     with open(str(base) + ".meta.json", "w") as f:  # → <OUT_DIR>/audio_set.meta.json
#         json.dump(meta, f, indent=2)
#
#     print(f"✅ Saved features: {base}.npz")
#     print(f"📁 Saved meta:     {base}.meta.json")
#
#     print(f"Output shape: {feats.shape}")
#
#     if grid_info:
#         print(
#             f"📊 Grid info: {grid_info['nf']}×{grid_info['nt']} patches, strides: ({grid_info['stride_f']:.1f}, {grid_info['stride_t']:.1f})")
#
#     if "time_axis" in payload:
#         print(
#             f"⏰ Time axis: {len(payload['time_axis'])} time points from {payload['time_axis'][0]:.3f}s to {payload['time_axis'][-1]:.3f}s")


# if __name__ == "__main__":
#     main()


# ⬇️ BEGIN ORIGINAL FILE CONTENT (retained) ⬇️
# [Everything from your original extract_audio_features_from_AST.py remains as-is]
# ast_level1_extractor_v2_dir_astmod
# (… your original imports and helper functions/classes are here …)

# ⬆️ END ORIGINAL FILE CONTENT (retained) ⬆️

# ======================
# Additions: Storehouse API (non-intrusive)
# ======================
from pathlib import Path
import tempfile, os, io, subprocess
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from transformers import ASTModel, ASTFeatureExtractor
import torchaudio
from torchaudio.transforms import Resample

def _atomic_torch_save(obj: dict, out_path: Path) -> None:
    """Atomic torch.save to avoid partial files on crash."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(out_path.parent), delete=False) as tf:
        tmp = Path(tf.name)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, out_path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise

## NEW fixed GPU code
class ASTAudioExtractor:
    def __init__(
            self,
            model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
            device: str | torch.device = "cuda",
            amp: bool = True,
            time_series: bool = True,
            token_pool: str = "none",
            sampling_rate: int = 16000,
            n_mels: int = 128,
            max_length: int = 1024,
            verbose: bool = False,
            default_save_dir: Optional[str] = None,
    ):
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

        self.use_amp = bool(amp)
        self.time_series = bool(time_series)
        self.token_pool = token_pool
        self.sr = sampling_rate
        self.n_mels = n_mels
        self.max_length = max_length
        self.verbose = verbose
        self.default_save_dir = Path(default_save_dir) if default_save_dir else None

        # Build HF frontend + backbone
        self.fe = ASTFeatureExtractor(
            sampling_rate=self.sr,
            n_mels=self.n_mels,
            do_normalize=True,
            max_length=self.max_length,
            return_attention_mask=False,
        )
        self.model = ASTModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
        # torch.set_grad_enabled(False)

    # ... [Keep all other methods unchanged] ...

    # ---- I/O helpers (demux & collate) ----
    def _demux_video_to_wav(self, video_path: str, mono: bool = True) -> np.ndarray:
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", video_path,
            "-f", "wav", "-acodec", "pcm_s16le",
            "-ar", str(self.sr),
            "-ac", "1" if mono else "2", "pipe:1"
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg error for '{video_path}': {p.stderr.decode()}")
        wav_bytes = io.BytesIO(p.stdout)
        wav, sr = torchaudio.load(wav_bytes)  # (C, N)
        if mono and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sr:
            wav = Resample(orig_freq=sr, new_freq=self.sr)(wav)
        return wav.squeeze(0).numpy()  # np.float32
    # def _demux_video_to_wav(self, video_path: str, mono: bool = True) -> np.ndarray:
    #     """
    #     Use torchaudio instead of ffmpeg for audio extraction
    #     """
    #     try:
    #         import torchaudio
    #
    #         # Load audio directly from video file
    #         wav, sr = torchaudio.load(video_path)
    #
    #         if mono and wav.shape[0] > 1:
    #             wav = wav.mean(dim=0, keepdim=True)
    #
    #         if sr != self.sr:
    #             from torchaudio.transforms import Resample
    #             resampler = Resample(orig_freq=sr, new_freq=self.sr)
    #             wav = resampler(wav)
    #
    #         return wav.squeeze(0).numpy()  # Convert to numpy array
    #
    #     except Exception as e:
    #         if self.verbose:
    #             print(f"⚠️ Audio extraction failed for {video_path}: {e}")
    #         # Return 5 seconds of silence as fallback
    #         duration = 5.0
    #         samples = int(duration * self.sr)
    #         return np.zeros(samples, dtype=np.float32)
    def _collate_wavs_to_inputs(self, wavs: List[np.ndarray]) -> torch.Tensor:
        out = self.fe(wavs, sampling_rate=self.sr, return_tensors="pt")
        return out["input_values"]  # (B, 1, n_mels, T) CPU

    # ---- forward + shaping ----
    def _forward_tokens(self, input_values: torch.Tensor) -> torch.Tensor:
        from torch import amp as _amp
        use_cuda_amp = (self.use_amp and self.device.type == "cuda")
        with torch.no_grad(), _amp.autocast(device_type=self.device.type, enabled=use_cuda_amp):            outputs = self.model(input_values=input_values.to(self.device, non_blocking=True))
        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        elif isinstance(outputs, tuple) and torch.is_tensor(outputs[0]):
            tokens = outputs[0]
        else:
            raise RuntimeError("AST forward did not return hidden-state tensor")
        return tokens.detach().cpu()  # (B, L, D)

    def _tokens_to_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        (B, L, D) -> default time-series (B, T', D) by collapsing freq.
        We remove the two leading specials: [CLS], [DIST].
        """
        if self.time_series:
            L = tokens.shape[1]  # embedded length incl. specials
            tokens_wo = tokens[:, 2:, :]  # drop [CLS] and [DIST]
            try:
                out = tokens_to_time_series_auto(tokens_wo, self.model, self.fe, L_incl_specials=L)
            except ValueError:
                # Safety fallback if this checkpoint only has 1 special:
                tokens_wo = tokens[:, 1:, :]
                out = tokens_to_time_series_auto(tokens_wo, self.model, self.fe, L_incl_specials=L)
            ts = out[0] if isinstance(out, tuple) else out
            return ts

        # non-time-series branch unchanged
        if self.token_pool == "none":
            return tokens
        if self.token_pool == "max":
            return tokens.max(dim=1).values
        return tokens.mean(dim=1)

    # ---- public API ----
    def extract_one(self, video_path: str,
                    save: bool = False,
                    save_dir: Optional[str] = None,
                    overwrite: bool = False) -> Optional[Dict[str, Any]]:
        p = Path(video_path)
        out_dir = Path(save_dir) if save_dir else (self.default_save_dir or p.parent / "ast_feats_pt")
        out_path = out_dir / f"{p.stem}.pt"
        if save and not overwrite and out_path.exists() and out_path.stat().st_size > 0:
            if self.verbose:
                print(f"[AST][skip] exists: {out_path}")
            return None

        # demux -> collate -> forward -> features
        wav = self._demux_video_to_wav(str(p))
        inputs = self._collate_wavs_to_inputs([wav])  # (1, 1, n_mels, T)
        tokens = self._forward_tokens(inputs)         # (1, L, D) CPU
        feats = self._tokens_to_features(tokens).squeeze(0)  # remove batch dim

        result = {"path": str(p), "features": feats, "shape": tuple(feats.shape)}
        if save:
            payload = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}
            _atomic_torch_save(payload, out_path)
        return result

    def extract_from_paths(self, paths: List[str],
                           save: bool = False,
                           save_dir: Optional[str] = None,
                           overwrite: bool = False) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        print("Audio Extraction stage reached!")
        for vp in paths:
            try:
                r = self.extract_one(vp, save=save, save_dir=save_dir, overwrite=overwrite)
                if r is not None:
                    out.append(r)
            except Exception as e:
                if self.verbose:
                    print(f"[AST][WARN] {vp}: {e}")
        return out
