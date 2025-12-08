#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
from pathlib import Path

# Optional: for PCA and plotting
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_npz_vectors(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    names, vecs = [], []
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            # Our extractor saved as {video_stem: vector}
            # Take the first (and only) key
            keys = list(data.keys())
            if not keys:
                print(f"[WARN] No keys in {f}, skipping.")
                continue
            k = keys[0]
            arr = data[k]
            # If temporal or list slipped in, reduce to mean for this quick check
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                # object array of arrays -> mean them
                arr = np.mean([a for a in arr.tolist()], axis=0)
            elif arr.ndim > 1:
                # e.g., (clips, C) -> mean over clips
                arr = arr.mean(axis=0)
            names.append(Path(f).stem)
            vecs.append(arr.astype(np.float32))
        except Exception as e:
            print(f"[ERROR] Failed to load {f}: {e}")
    if not vecs:
        raise RuntimeError(f"No usable .npz feature files found in: {out_dir}")
    X = np.stack(vecs, axis=0)  # (N, D)
    return names, X

def cosine_similarity_matrix(X):
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    return Xn @ Xn.T  # (N, N)

def print_similarity_report(names, S):
    N = len(names)
    # off-diagonal similarities
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    print("\n=== Cosine similarity (off-diagonal) ===")
    print(f"min: {np.nanmin(off):.4f} | max: {np.nanmax(off):.4f} | mean: {np.nanmean(off):.4f}")

    # top-3 most similar pairs (excluding identity)
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((S[i, j], i, j))
    pairs.sort(reverse=True, key=lambda t: t[0])

    print("\nTop-3 most similar pairs:")
    for score, i, j in pairs[:3]:
        print(f"  {names[i]}  <->  {names[j]}   cos={score:.4f}")

def pca_plot(names, X, out_path="embeddings_pca.png"):
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)  # (N,2)

    plt.figure(figsize=(6, 5))
    plt.scatter(Z[:,0], Z[:,1])
    for i, n in enumerate(names):
        plt.annotate(n, (Z[i,0], Z[i,1]), xytext=(3,3), textcoords="offset points", fontsize=8)
    plt.title("PCA (2D) of MViTv2 embeddings")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved PCA scatter to: {out_path}")

def main():
    out_dir = "/Users/abhishekgupte_macbookpro/PycharmProjects/new_pitch_estimator_project/mvitv2_pipeline_feature_Extraction/mvitv2_torchvision/features_out"
    names, X = load_npz_vectors(out_dir)
    print(f"Loaded {len(names)} embeddings with dim={X.shape[1]}")

    S = cosine_similarity_matrix(X)
    print_similarity_report(names, S)
    pca_plot(names, X, out_path=os.path.join(out_dir, "embeddings_pca.png"))

if __name__ == "__main__":
    main()