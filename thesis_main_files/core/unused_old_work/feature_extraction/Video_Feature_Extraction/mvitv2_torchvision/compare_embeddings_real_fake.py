#!/usr/bin/env python3
"""
Compare embeddings of real vs fake video using MViTv2 (TorchVision).
CPU-only, shows plots in real time.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_video

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Video loading & preprocessing
# -------------------------------
def read_video_centerclip(path, target_fps=25, num_frames=16, resize=(224,224)):
    vframes, aframes, info = read_video(str(path), pts_unit="sec")
    fps = info.get("video_fps", target_fps)

    total = vframes.shape[0]
    if total == 0:
        raise RuntimeError(f"No frames in {path}")

    idxs = np.linspace(0, total-1, num_frames).astype(int)
    vframes = vframes[idxs]  # sample evenly

    transform = T.Compose([
        T.Resize(resize),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    # (T,H,W,C) -> (T,C,H,W)
    clip = vframes.permute(0,3,1,2)
    clip = transform(clip)
    clip = clip.permute(1,0,2,3)  # (C,T,H,W)
    return clip.unsqueeze(0)      # (1,C,T,H,W)


# -------------------------------
# Embedding extraction
# -------------------------------
def extract_embeddings(model, video_paths, label, device="cpu"):
    embeddings, labels = [], []
    for p in video_paths:
        try:
            clip = read_video_centerclip(p).to(device)
            with torch.no_grad():
                feats = model.forward_head(model.forward_features(clip))
            embeddings.append(feats.squeeze(0).cpu().numpy())
            labels.append(label)
            print(f"✓ Processed {p.name}")
        except Exception as e:
            print(f"✗ Failed {p.name}: {e}")
    return embeddings, labels


# -------------------------------
# Main
# -------------------------------
def main(args):
    device = "cpu"  # force CPU

    # Load model
    weights = torch.hub.load("pytorch/vision", "mvit_v2_s", weights="KINETICS400_V1")
    model = weights.to(device).eval()

    # Gather files
    real_dir = Path(args.root_dir) / "eval_real"
    fake_dir = Path(args.root_dir) / "eval_fake"
    real_videos = list(real_dir.glob("*.mp4"))
    fake_videos = list(fake_dir.glob("*.mp4"))

    print(f"Found {len(real_videos)} real, {len(fake_videos)} fake video")

    # Extract
    real_emb, real_lbl = extract_embeddings(model, real_videos, "real", device)
    fake_emb, fake_lbl = extract_embeddings(model, fake_videos, "fake", device)

    all_emb = np.vstack(real_emb + fake_emb)
    all_lbl = np.array(real_lbl + fake_lbl)

    # PCA scatter
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_emb)

    plt.figure(figsize=(6,5))
    for label, color in [("real","green"),("fake","red")]:
        idx = np.where(all_lbl==label)
        plt.scatter(reduced[idx,0], reduced[idx,1], c=color, label=label, alpha=0.7)
    plt.legend()
    plt.title("PCA Scatter of Embeddings (Real vs Fake)")
    plt.show()

    # Similarity heatmap
    sim = cosine_similarity(all_emb)
    plt.figure(figsize=(6,5))
    plt.imshow(sim, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Cosine similarity")
    plt.title("Cosine Similarity Heatmap")
    plt.show()

    # Bars (real vs fake avg sim)
    real_mean = np.mean(cosine_similarity(real_emb))
    fake_mean = np.mean(cosine_similarity(fake_emb))
    cross_mean = np.mean(cosine_similarity(np.vstack(real_emb), np.vstack(fake_emb)))

    plt.figure(figsize=(5,4))
    plt.bar(["real-real","fake-fake","real-fake"],
            [real_mean, fake_mean, cross_mean],
            color=["green","red","purple"])
    plt.ylabel("Avg cosine similarity")
    plt.title("Embedding similarity")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root dir with eval_real/ and eval_fake/")
    args = parser.parse_args()
    main(args)
