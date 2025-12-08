#!/usr/bin/env python3
import os, json, numpy as np
import matplotlib.pyplot as plt

def cosine(a, b, axis=-1, eps=1e-9):
    a_n = a / (np.linalg.norm(a, axis=axis, keepdims=True) + eps)
    b_n = b / (np.linalg.norm(b, axis=axis, keepdims=True) + eps)
    return np.sum(a_n * b_n, axis=axis)

def main(npz_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    z = np.load(npz_path, allow_pickle=True)
    X = z["features"]          # (B, T=101, C=768)
    paths = z["paths"]         # (B,)
    t = z["time_axis"]         # (T,)
    B, T, C = X.shape

    # --- Basic variation metrics ---
    temporal_var = np.var(X, axis=1).mean(axis=1)   # var over time, mean over dims -> (B,)
    embedding_var = np.var(X, axis=2).mean(axis=1)  # var over dims, mean over time -> (B,)

    # Clip-level mean embeddings and pairwise cosine
    clip_means = X.mean(axis=1)                     # (B, C)
    pair_cos = np.zeros((B, B))
    time_align_cos = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            pair_cos[i, j] = cosine(clip_means[i], clip_means[j])
            time_align_cos[i, j] = cosine(X[i], X[j], axis=1).mean()  # mean cosine over time

    # --- Save JSON summary ---
    summary = {
        "shapes": {"features": list(X.shape), "time_axis": list(t.shape)},
        "paths": [str(p) for p in paths],
        "temporal_variance_mean_per_clip": temporal_var.tolist(),
        "embedding_variance_mean_per_clip": embedding_var.tolist(),
        "pairwise_clip_mean_cosine": pair_cos.tolist(),
        "pairwise_time_aligned_cosine_mean_over_time": time_align_cos.tolist(),
    }
    with open(os.path.join(out_dir, "variation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---
    # 1) Temporal L2 norms per clip (||x_t|| over time, plotted vs seconds)
    plt.figure()
    for i in range(B):
        norms = np.linalg.norm(X[i], axis=1)  # (T,)
        plt.plot(t, norms, label=os.path.basename(paths[i]))
    plt.xlabel("Time (s)")
    plt.ylabel("L2 norm of embedding")
    plt.title("Temporal embedding norms per clip")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "temporal_norms.png"), dpi=150)

    # 2) Heatmaps per clip (clipped for visibility)
    for i in range(B):
        plt.figure()
        img = np.clip(X[i], -3.0, 3.0)
        plt.imshow(img, aspect="auto")
        plt.xlabel(f"Embedding dim (C={C})")
        plt.ylabel("Time step (0..100)")
        plt.title(f"Feature matrix (clipped) — {os.path.basename(paths[i])}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"matrix_{i+1}.png"), dpi=150)

    print("[OK] Saved variation_summary.json and plots to:", out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_file", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.npz_file, args.out_dir)
