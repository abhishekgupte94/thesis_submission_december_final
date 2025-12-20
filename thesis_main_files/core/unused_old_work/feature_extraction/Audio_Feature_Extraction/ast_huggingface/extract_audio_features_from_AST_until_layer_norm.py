# ast_level1_extractor_v2_dir_astmodel_LN.py
# Directory-based AST feature extractor (no CSV).
# - Supports AUDIO files or VIDEO files (demux audio via ffmpeg).
# - Uses ASTModel (backbone only).
# - Extracts features at LayerNorm after Attention residual (pre-FFN) on the FINAL block.
# - Optional: reshape to time-series (avg over freq patches) and/or pool.
# - Enhanced with automatic grid evaluation_for_detection_model and time axis generation.
# - Changed: --out is treated as a directory; saves <out>/audio_set.npz and <out>/audio_set.meta.json

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


def _cfg_get(cfg, *names, default=None):
    for n in names:
        if hasattr(cfg, n):
            v = getattr(cfg, n)
            if v is not None:
                return v
        if isinstance(cfg, dict) and n in cfg and cfg[n] is not None:
            return cfg[n]
    return default

def count_specials(embeddings_module) -> int:
    n = 0
    if hasattr(embeddings_module, "cls_token") and embeddings_module.cls_token is not None:
        n += 1
    if hasattr(embeddings_module, "distillation_token") and embeddings_module.distillation_token is not None:
        n += 1
    if hasattr(embeddings_module, "distill_token") and embeddings_module.distill_token is not None:
        n += 1
    return n

def ast_grid_from_config(model, feat_extractor, L_incl_specials: Optional[int] = None) -> Dict[str, float]:
    cfg = model.config
    patch_size = _cfg_get(cfg, "patch_size", default=16)
    if isinstance(patch_size, int):
        pf = pt = int(patch_size)
    else:
        pf, pt = int(patch_size[0]), int(patch_size[1])

    F = getattr(feat_extractor, "num_mel_bins", 128)
    T = getattr(feat_extractor, "max_length", 1024)

    stride_f = _cfg_get(cfg, "fstride", "freq_stride", "stride_f", default=None)
    stride_t = _cfg_get(cfg, "tstride", "time_stride", "stride_t", default=None)
    if stride_f is None and stride_t is None:
        s = _cfg_get(cfg, "stride", default=None)
        if s is not None:
            if isinstance(s, (tuple, list)) and len(s) == 2:
                stride_f, stride_t = int(s[0]), int(s[1])
            elif isinstance(s, int):
                stride_f = stride_t = int(s)

    specials = count_specials(model.embeddings)

    if stride_f is not None and stride_t is not None:
        nf = int(math.floor((F - pf) / float(stride_f)) + 1)
        nt = int(math.floor((T - pt) / float(stride_t)) + 1)
        N = nf * nt
        return {"nf": nf, "nt": nt, "pf": pf, "pt": pt,
                "stride_f": float(stride_f), "stride_t": float(stride_t),
                "specials": specials, "N": N, "F": F, "T": T}

    if L_incl_specials is None:
        raise ValueError("Stride not in config; pass L_incl_specials to infer grid from tokens.")
    N = int(L_incl_specials) - int(specials)

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
        sf = (F - pf) / (nf - 1) if nf > 1 else float("inf")
        st = (T - pt) / (nt - 1) if nt > 1 else float("inf")

        def ok(x): return x > 0 and x <= max(pf, pt) + 1e-6
        if ok(sf) and ok(st):
            int_err = abs(sf - round(sf)) + abs(st - round(st))
            sym_err = abs(sf - st)
            score = int_err + 0.25 * sym_err
            cand = (score, nf, nt, sf, st)
            best = cand if (best is None or score < best[0]) else best

    if best is None:
        raise RuntimeError(f"Could not infer grid for N={N}, F={F}, T={T}, pf={pf}, pt={pt}.")

    _, nf, nt, sf, st = best
    return {"nf": int(nf), "nt": int(nt), "pf": pf, "pt": pt,
            "stride_f": float(sf), "stride_t": float(st),
            "specials": specials, "N": int(N), "F": F, "T": T}

def tokens_to_time_series_auto(tokens: torch.Tensor, model, feat_extractor,
                               *, L_incl_specials: Optional[int] = None,
                               sample_rate: int = 16000,
                               hop_length: int = 160,
                               return_time_axis: bool = False,
                               center: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    info = ast_grid_from_config(model, feat_extractor, L_incl_specials=L_incl_specials)
    nf, nt, pt = info["nf"], info["nt"], info["pt"]

    B, N, C = tokens.shape
    if nf * nt != N:
        raise ValueError(f"N={N} tokens does not match nf*nt={nf*nt}")

    grid = tokens.view(B, nf, nt, C)
    ts = grid.mean(dim=1)

    if not return_time_axis:
        return ts, None, info

    stride_frames = info["stride_t"]
    time_per_patch = (stride_frames * hop_length) / float(sample_rate)
    offset = 0.0
    if center:
        patch_duration = (pt * hop_length) / float(sample_rate)
        offset = 0.5 * patch_duration
    t = torch.arange(nt, device=tokens.device, dtype=tokens.dtype) * time_per_patch + offset
    return ts, t, info


class DirAudioDataset(Dataset):
    def __init__(self, data_dir: str, exts: List[str], source_type: str = "video",
                 target_sr: int = 16000, mono: bool = True,
                 recursive: bool = True, verbose: bool = False):
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

    def _extract_waveform_from_video(self, video_path: str) -> torch.Tensor:
        cmd = ["ffmpeg", "-v", "error", "-i", video_path,
               "-f", "wav", "-acodec", "pcm_s16le",
               "-ar", str(self.target_sr), "-ac", "1", "pipe:1"]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg error for '{video_path}': {p.stderr.decode()}")
        wav_bytes = io.BytesIO(p.stdout)
        wav, sr = torchaudio.load(wav_bytes)
        if self.mono and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = Resample(orig_freq=sr, new_freq=self.target_sr)(wav)
        return wav.squeeze(0)

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
        return {"input_values": out["input_values"], "paths": [b["path"] for b in batch]}


def ast_embed(backbone: ASTModel, pixel_values: torch.Tensor) -> torch.Tensor:
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
def ast_pre_ffn_layernorm_tokens(ast_backbone: ASTModel, pixel_values: torch.Tensor,
                                 remove_cls: bool = True) -> torch.Tensor:
    x = ast_embed(ast_backbone, pixel_values)
    layers = getattr(ast_backbone.encoder, "layer",
             getattr(ast_backbone.encoder, "layers", None))
    if layers is None:
        raise AttributeError("AST backbone has no encoder layers collection")
    last_ln_after_attn = None
    for block in layers:
        x_ln1 = block.layernorm_before(x)
        attn_out = block.attention(x_ln1)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        y = x + attn_out
        y_ln2 = block.layernorm_after(y)
        last_ln_after_attn = y_ln2
        intermediate_out = block.intermediate(y_ln2)
        x = block.output(intermediate_out, y)
    tokens = last_ln_after_attn
    if tokens is None:
        raise RuntimeError("Failed to capture LayerNorm-after-attention tensor.")
    if remove_cls:
        tokens = tokens[:, 1:, :]
    return tokens


def pool_tokens(tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if mode == "mean":
        return tokens.mean(dim=1)
    if mode == "max":
        return tokens.max(dim=1).values
    raise ValueError("pool mode must be 'mean' or 'max'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--exts", default=".mp4,.mov,.mkv,.avi,.wav,.flac,.mp3")
    ap.add_argument("--source-type", choices=["video", "audio"], default="video")
    ap.add_argument("--recursive", choices=["yes", "no"], default="no")
    ap.add_argument("--model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--num-mel-bins", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--time-series", choices=["yes", "no"], default="no")
    ap.add_argument("--token-pool", choices=["none", "mean", "max"], default="mean")
    ap.add_argument("--debug-tokens", action="store_true")
    ap.add_argument("--return-time-axis", choices=["yes", "no"], default="no")
    ap.add_argument("--center-time", choices=["yes", "no"], default="no")
    ap.add_argument("--hop-length", type=int, default=160)
    ap.add_argument("--out", required=True,
                    help="Output directory (files saved as <out>/audio_set.npz and <out>/audio_set.meta.json)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    recursive = (args.recursive.lower() == "yes")
    return_time_axis = (args.return_time_axis.lower() == "yes")
    center_time = (args.center_time.lower() == "yes")
    debug_tokens = args.debug_tokens

    ds = DirAudioDataset(args.data_dir, exts, args.source_type,
                         target_sr=args.sr, recursive=recursive, verbose=args.verbose)
    fe = ASTFeatureExtractor(sampling_rate=args.sr, num_mel_bins=args.num_mel_bins,
                             max_length=args.max_length, do_normalize=True)
    collate = ASTCollator(fe=fe, target_sr=args.sr)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"), collate_fn=collate, drop_last=False)

    cfg = ASTConfig.from_pretrained(args.model)
    cfg.num_mel_bins = args.num_mel_bins
    cfg.max_length = args.max_length
    if not hasattr(cfg, "stride") and not hasattr(cfg, "freq_stride"):
        cfg.stride = 16

    model = ASTModel.from_pretrained(args.model, config=cfg, ignore_mismatched_sizes=True).to(device).eval()

    feats_list, paths_list, time_axes_list = [], [], []
    grid_info = None

    with torch.no_grad():
        for batch in tqdm(dl, desc="Extracting (LN-after-Attn, pre-FFN)", ncols=100):
            x = batch["input_values"].to(device)
            tokens_with_specials = ast_pre_ffn_layernorm_tokens(model, x, remove_cls=False)
            specials_count = count_specials(model.embeddings)
            tokens = tokens_with_specials[:, specials_count:, :]

            if args.time_series == "yes":
                ts, time_axis, info = tokens_to_time_series_auto(
                    tokens, model, fe,
                    L_incl_specials=tokens_with_specials.shape[1],
                    sample_rate=args.sr, hop_length=args.hop_length,
                    return_time_axis=return_time_axis, center=center_time)
                rep = ts
                if grid_info is None:
                    grid_info = info
                if time_axis is not None:
                    time_axes_list.append(time_axis.cpu().numpy())
            else:
                rep = tokens

            if args.token_pool != "none":
                rep = pool_tokens(rep, mode=args.token_pool)

            feats_list.append(rep.cpu().float().numpy())
            paths_list.extend(batch["paths"])

    feats = np.concatenate(feats_list, axis=0)
    payload = {"features": feats, "paths": np.array(paths_list)}
    if time_axes_list:
        payload["time_axis"] = time_axes_list[0]

    meta = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "exts": exts,
        "recursive": recursive,
        "source_type": args.source_type,
        "model": args.model,
        "sr": args.sr,
        "num_mel_bins": args.num_mel_bins,
        "max_length": args.max_length,
        "hop_length": args.hop_length,
        "hidden_size": int(model.config.hidden_size),
        "time_series": args.time_series,
        "token_pool": args.token_pool,
        "return_time_axis": return_time_axis,
        "center_time": center_time,
        "fp16": bool(args.fp16 and device.type == "cuda"),
        "feature_level": "LN-after-Attn (pre-FFN) on final block",
        "output_shape": list(feats.shape),
    }
    if grid_info is not None:
        meta["grid_info"] = grid_info

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "audio_set"

    np.savez_compressed(str(base) + ".npz", **payload)
    with open(str(base) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved features: {base}.npz")
    print(f"Saved meta:     {base}.meta.json")
    print(f"Output shape: {feats.shape}")


if __name__ == "__main__":
    main()
