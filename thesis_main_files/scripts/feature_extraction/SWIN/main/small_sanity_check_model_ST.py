#!/usr/bin/env python
"""
Small sanity check for Swin2D backbone used as a token generator.

Verifies:
1) Forward_features runs
2) Output tokens have shape (B, S, D)
3) Gradients flow into patch embedding weights
"""

import time
import torch

# ---------------------------------------------------------------------
# Import your Swin2D builder
# (adjust import path if needed)
# ---------------------------------------------------------------------
from build_swin2d import build_swin2d_backbone, BuildSwin2DConfig


def main():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # Config (matches what you described)
    # ------------------------------------------------------------
    cfg = BuildSwin2DConfig(
        img_size=(96, 64),     # mel spectrogram size
        in_chans=1,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=8
        # patch_size=2
    )

    model = build_swin2d_backbone(cfg).to(device)
    model.train()

    # ------------------------------------------------------------
    # Dummy mel input: B x 1 x F x T
    # ------------------------------------------------------------
    B = 1
    x = torch.randn(B, 1, 96, 64, device=device, requires_grad=True)

    start = time.time()

    # ------------------------------------------------------------
    # Forward (tokens only)
    # ------------------------------------------------------------
    tokens = model.forward_features(x)

    print(f"[OK] tokens shape: {tuple(tokens.shape)}")

    # Expect: (B, S, D)
    assert tokens.ndim == 3, "Expected B x S x D tokens"

    # ------------------------------------------------------------
    # Dummy loss (keep it trivial)
    # ------------------------------------------------------------
    loss = tokens.mean()
    loss.backward()

    # ------------------------------------------------------------
    # Gradient check (critical)
    # ------------------------------------------------------------
    grad_checks = [
        "backbone.patch_embed.proj.weight",
        "backbone.layers.0.blocks.0.attn.qkv.weight",
        "backbone.norm.weight",
    ]

    found_grad = False
    for name, param in model.named_parameters():
        if name in grad_checks and param.grad is not None:
            g = param.grad
            print(
                f"[OK] grad: {name} | "
                f"max_abs={g.abs().max().item():.3e} | "
                f"norm={g.norm().item():.3e}"
            )
            found_grad = True

    assert found_grad, "No gradients found in backbone!"

    elapsed = time.time() - start
    print(f"[DONE] Swin2D backbone sanity passed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
