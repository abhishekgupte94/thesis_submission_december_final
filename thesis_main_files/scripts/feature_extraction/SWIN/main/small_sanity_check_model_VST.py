#!/usr/bin/env python3
from __future__ import annotations
# from __future__ import annotations
import time
import torch
from pathlib import Path
import sys

VST_ROOT = Path(__file__).resolve().parents[4] / "external" / "Video-Swin-Transformer"
# ^ adjust parents[...] so it reaches thesis_main_files/ then /external/Video-Swin-Transformer

# Make sure the VST repo root is FIRST so its ./mmaction package wins.
sys.path.insert(0, str(VST_ROOT))
from scripts.feature_extraction.SWIN.main.build_swin3d import (
    build_swin3d_backbone,
    BuildSwin3DConfig,
)



from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def decode_mp4_to_bcthw_float(
    mp4_path,
    *,
    num_frames=None,
    resize_hw=None,
    rgb=True,
    normalize_01=True,     # True => /255
    device=None,
) -> torch.Tensor:
    import cv2
    import numpy as np
    import torch

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_hw is not None:
                H, W = resize_hw
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {mp4_path}")

    arr = np.stack(frames, axis=0)  # (T,H,W,3) uint8
    x = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()  # (C,T,H,W)

    # --- IMPORTANT: convert dtype for conv3d ---
    x = x.to(torch.float32)
    if normalize_01:
        x = x / 255.0

    # force T if requested
    if num_frames is not None:
        C, T, H, W = x.shape
        if T > num_frames:
            idx = torch.linspace(0, T - 1, steps=num_frames).round().to(torch.long)
            x = x[:, idx]
        elif T < num_frames:
            pad_t = num_frames - T
            last = x[:, -1:, :, :].expand(C, pad_t, H, W)
            x = torch.cat([x, last], dim=1)

    x = x.unsqueeze(0)  # (1,C,T,H,W)
    if device is not None:
        x = x.to(device, non_blocking=True)
    return x



def main():
    t0 = time.time()

    # Keep this small to avoid hammering your Mac
    cfg = BuildSwin3DConfig(
        pretrained=None,         # scratch
        pretrained2d=True,        # irrelevant if pretrained=None, but matches signature
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=32,
        depths=(1, 1, 2, 1),
        num_heads=(1, 2, 4, 8),
        window_size=(2, 4, 4),
        drop_path_rate=0.1,
        patch_norm=False,
        frozen_stages=-1,
        use_checkpoint=False,
        out="5d",                 # returns (B,C,T',H',W') like the backbone
    )

    x = decode_mp4_to_bcthw_float("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/video_files/LAV_DF/video/000469.mp4")
    model = build_swin3d_backbone(cfg)
    model.train()

    # Tiny input: (B, C, T, H, W)
    # x = torch.randn(1, 3, 8, 224, 224, dtype=torch.float32, requires_grad=True)

    y = model.forward_features(x)
    print(f"[OK] forward output shape: {tuple(y.shape)}")

    loss = y.mean()
    loss.backward()

    # Check grads in early + late params
    checks = [
        "backbone.patch_embed.proj.weight",
        "backbone.layers.0.blocks.0.attn.qkv.weight",
        "backbone.layers.3.blocks.0.attn.qkv.weight",  # last stage (if depths allow)
        "backbone.norm.weight",
    ]

    name_to_param = dict(model.named_parameters())

    for k in checks:
        p = name_to_param.get(k, None)
        if p is None:
            print(f"[WARN] param not found: {k}")
            continue
        if p.grad is None:
            print(f"[FAIL] no grad: {k}")
        else:
            print(f"[OK] grad: {k} | max_abs={p.grad.abs().max().item():.3e} | norm={p.grad.norm().item():.3e}")

    # Confirm at least one parameter received gradients
    got_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            got_grad = True
            print(f"[OK] Grad flowed into: {name} | grad_norm={p.grad.norm().item():.6f}")
            break

    if not got_grad:
        raise RuntimeError("No parameter gradients found — gradient flow did not reach the backbone.")

    dt = time.time() - t0
    print(f"[DONE] Swin3D backbone sanity passed in {dt:.2f}s")


if __name__ == "__main__":
    main()
