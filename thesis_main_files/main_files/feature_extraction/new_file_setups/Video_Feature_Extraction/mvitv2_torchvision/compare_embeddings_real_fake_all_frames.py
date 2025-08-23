#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal embedding comparison with DTW (CPU, live plots)
- Processes the ENTIRE video (resampled to target_fps)
- Slides a 16-frame window with configurable stride across all frames
- One embedding per clip via MViTv2-S (TorchVision) using a forward hook at model.norm
- Compares videos with DTW (cosine distance)
- Live plots: DTW heatmap, PCA trajectories, and a summary bar chart

Expected directory:
<root_dir>/
  eval_real/*.mp4
  eval_fake/*.mp4
"""

import sys, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# ---- model-safe constants ----
MVIT_SAFE_T = 16
MVIT_SAFE_HW = (224, 224)  # (H, W)

# ----------------- helpers -----------------
def set_cpu_sanity():
    """Calm CPU thread usage on macOS."""
    cv2.setNumThreads(0)
    torch.set_num_threads(1)

def parse_args():
    p = argparse.ArgumentParser("REAL vs FAKE — temporal DTW (CPU, live plots)")
    p.add_argument("--root_dir", required=True, type=str, help="Folder with eval_real/ and eval_fake/")
    p.add_argument("--target_fps", default=25.0, type=float, help="Resampled FPS")
    p.add_argument("--stride", default=8, type=int, help="Stride in frames for the 16-frame window")
    p.add_argument("--limit", default=0, type=int, help="Optional total cap (balanced across classes)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# ----------------- video I/O -----------------
def sample_full_video_cv2(path: Path, target_fps: float) -> np.ndarray:
    """
    Read whole video and resample to target_fps by nearest-frame selection.
    Returns RGB uint8 array (T, H, W, 3).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames in {path}")

    duration = total / max(src_fps, 1e-6)
    step = 1.0 / float(target_fps)
    target_ts = np.arange(0.0, duration, step)
    if len(target_ts) == 0:
        cap.release()
        raise RuntimeError(f"No target timestamps for {path}")

    def ts_of(i): return i / src_fps

    frames = []
    ret, idx, tgt_i = True, 0, 0
    while ret and tgt_i < len(target_ts):
        ret, frame_bgr = cap.read()
        if not ret: break
        # choose nearest actual frame to current target timestamp
        t = ts_of(idx)
        nearer = True
        if (idx + 1) < total:
            nearer = abs(t - target_ts[tgt_i]) <= abs(ts_of(idx + 1) - target_ts[tgt_i])
        if nearer:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            tgt_i += 1
        idx += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"Resampling produced 0 frames for {path}")
    return np.stack(frames, axis=0)  # (T,H,W,3)

def build_starts(T: int, clip_len: int, stride: int) -> List[int]:
    if T <= clip_len: return [0]
    starts = list(range(0, T - clip_len + 1, max(1, stride)))
    if starts[-1] != T - clip_len:
        starts.append(T - clip_len)
    return starts

# --------------- model & preprocess ---------------
@torch.inference_mode()
def load_model_with_hook_cpu():
    """Load TorchVision MViT-V2-S with Kinetics-400 weights, hook tokens at model.norm."""
    weights = torchvision.models.video.MViT_V2_S_Weights.KINETICS400_V1
    model = torchvision.models.video.mvit_v2_s(weights=weights).eval()  # CPU
    HOOK = {"tokens": None}
    def norm_hook(_, __, output):
        HOOK["tokens"] = output.detach()
    model.norm.register_forward_hook(norm_hook)
    # Warmup (stabilize kernels)
    _ = model(torch.zeros(1, 3, MVIT_SAFE_T, MVIT_SAFE_HW[0], MVIT_SAFE_HW[1]))
    return model, HOOK

def clip_to_tensor_cpu(frames_np: np.ndarray) -> torch.Tensor:
    """
    frames_np: (T,H,W,3) uint8 RGB, T==MVIT_SAFE_T
    returns: (1,3,T,224,224) float32 normalized, on CPU
    """
    x = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T,3,H,W)
    x = F.interpolate(x.float(), size=MVIT_SAFE_HW, mode="bilinear", align_corners=False)
    x = x.permute(1, 0, 2, 3).contiguous()              # (3,T,224,224)
    x = x / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1)
    x = (x - mean) / std
    return x.unsqueeze(0)                                # (1,3,T,H,W)

@torch.inference_mode()
def clip_embedding(model, hook, clip_np: np.ndarray) -> np.ndarray:
    """Embed one 16-frame clip → (C,) by mean-pooling tokens captured at model.norm."""
    if clip_np.shape[0] < MVIT_SAFE_T:
        # pad with last frame
        last = clip_np[-1] if clip_np.size else np.zeros((MVIT_SAFE_HW[0], MVIT_SAFE_HW[1], 3), dtype=np.uint8)
        while clip_np.shape[0] < MVIT_SAFE_T:
            clip_np = np.concatenate([clip_np, last[None, ...]], axis=0)
    batch = clip_to_tensor_cpu(clip_np)
    _ = model(batch)                 # tokens captured by hook
    toks = hook["tokens"]            # (1, N, C)
    if toks is None:
        raise RuntimeError("norm hook failed to capture tokens.")
    return toks.mean(dim=1).squeeze(0).cpu().numpy()  # (C,)

def embed_sequence_for_video(model, hook, path: Path, target_fps: float, stride: int, verbose=False) -> np.ndarray:
    """Full-video sequence of clip embeddings: (L, C)"""
    frames = sample_full_video_cv2(path, target_fps)     # (T,H,W,3)
    starts = build_starts(frames.shape[0], MVIT_SAFE_T, stride)
    seq = []
    for s in starts:
        clip_np = frames[s:s + MVIT_SAFE_T]
        emb = clip_embedding(model, hook, clip_np)
        seq.append(emb)
    seq = np.stack(seq, axis=0)                          # (L,C)
    if verbose:
        print(f"  {path.name}: frames={frames.shape[0]}, clips={len(starts)}, emb_dim={seq.shape[1]}")
    return seq

# --------------- DTW (cosine) ---------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb + 1e-8))

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Classic O(n*m) DTW with cosine distance; length-normalized score."""
    n, m = len(seq1), len(seq2)
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cosine_distance(seq1[i - 1], seq2[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))

# --------------- dataset listing ---------------
def list_videos_balanced(root_dir: Path, limit: int = 0) -> Tuple[List[Path], List[str]]:
    real = sorted((root_dir / "eval_real").glob("*.mp4"))
    fake = sorted((root_dir / "eval_fake").glob("*.mp4"))
    if limit and limit > 0:
        half = max(1, limit // 2)
        real = real[:half]
        fake = fake[:limit - len(real)]
    files = real + fake
    labels = ["real"] * len(real) + ["fake"] * len(fake)
    return files, labels

# --------------- plotting ---------------
def pca_fit_transform(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=0).fit_transform(X)
    except Exception:
        # fallback SVD
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:n_components].T

def plot_dtw_heatmap(D: np.ndarray, names: List[str]):
    plt.figure(figsize=(6, 5))
    plt.imshow(D, interpolation="nearest")
    plt.xticks(range(len(names)), names, rotation=75, ha="right", fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.colorbar(fraction=0.046, pad=0.04, label="DTW (cosine)")
    plt.title("DTW distance heatmap (clip-sequence embeddings)")
    plt.tight_layout()
    plt.show()

def plot_pca_trajectories(seqs: List[np.ndarray], labels: List[str], names: List[str]):
    X = np.concatenate(seqs, axis=0)      # (sum L, C)
    Z = pca_fit_transform(X, n_components=2)
    idx = 0
    plt.figure(figsize=(6, 5))
    for seq, label, name in zip(seqs, labels, names):
        L = len(seq)
        traj = Z[idx:idx + L]; idx += L
        color = "tab:blue" if label == "real" else "tab:red"
        plt.plot(traj[:, 0], traj[:, 1], marker="o", linewidth=1.5, markersize=3,
                 color=color, alpha=0.9, label=name)
        plt.annotate(f"{name}:0", (traj[0, 0], traj[0, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        plt.annotate(f"{name}:{L-1}", (traj[-1, 0], traj[-1, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.title("PCA trajectories — real (blue) vs fake (red)")
    plt.tight_layout()
    plt.show()

def plot_similarity_bars(D: np.ndarray, labels: List[str]):
    idx_r = [i for i, l in enumerate(labels) if l == "real"]
    idx_f = [i for i, l in enumerate(labels) if l == "fake"]

    def mean_offdiag(idxs):
        if len(idxs) < 2: return np.nan
        vals = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                vals.append(D[idxs[i], idxs[j]])
        return float(np.mean(vals)) if vals else np.nan

    def mean_cross(a, b):
        vals = [D[i, j] for i in a for j in b]
        return float(np.mean(vals)) if vals else np.nan

    intra_real = mean_offdiag(idx_r)
    intra_fake = mean_offdiag(idx_f)
    inter_rf   = mean_cross(idx_r, idx_f)

    plt.figure(figsize=(5, 4))
    cats = ["Intra REAL (DTW↓)", "Intra FAKE (DTW↓)", "REAL↔FAKE (DTW↓)"]
    vals = [intra_real, intra_fake, inter_rf]
    plt.bar(cats, vals)
    plt.ylabel("Mean DTW distance (lower is closer)")
    plt.title("Temporal similarity summary")
    for i, v in enumerate(vals):
        if not np.isnan(v):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.show()

# ----------------- main -----------------
def main():
    args = parse_args()
    set_cpu_sanity()

    root = Path(args.root_dir)
    files, labels = list_videos_balanced(root, args.limit)
    if not files:
        print(f"No videos under {root}/{{eval_real,eval_fake}}"); sys.exit(1)
    names = [p.stem for p in files]
    if args.verbose:
        print(f"Found {len(files)} videos → real={labels.count('real')} fake={labels.count('fake')}")

    model, HOOK = load_model_with_hook_cpu()

    # Build per-video sequences of embeddings
    seqs = []
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p.name} ({labels[i-1]})")
        seq = embed_sequence_for_video(model, HOOK, p, args.target_fps, args.stride, verbose=args.verbose)
        seqs.append(seq)

    # DTW distance matrix
    N = len(seqs)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if j < i:
                D[i, j] = D[j, i]
            elif j == i:
                D[i, j] = 0.0
            else:
                D[i, j] = dtw_distance(seqs[i], seqs[j])

    # Live plots
    plot_dtw_heatmap(D, names)
    plot_pca_trajectories(seqs, labels, names)
    plot_similarity_bars(D, labels)

if __name__ == "__main__":
    main()
