#!/usr/bin/env python
"""
sanity_audio_preprocessor_swin_tiny_trainstep.py

Goal:
- Run AudioPreprocessorNPV -> mel -> x (B,1,H,W)
- Build your Swin2D backbone via build_swin2d_backbone()
- Robustly extract features even if model(x) returns None
- Run a tiny SSL-ish train scenario:
    view1, view2 -> backbone -> embedding -> projection -> cosine loss
- Verify gradients flow into the Swin backbone
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Repo path helper (same as your script)
# =============================================================================
def _add_repo_paths_to_syspath() -> None:
    this_file = Path(__file__).resolve()

    for p in [this_file.parent, *this_file.parents]:
        if p.name == "thesis_main_files":
            repo_root = p.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return

    for p in [this_file.parent, *this_file.parents]:
        if (p / "scripts").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return

    raise RuntimeError(
        "Could not locate repo root automatically. "
        "Place this sanity script under your repo (preferably under thesis_main_files/), "
        "or adjust _add_repo_paths_to_syspath() to point at the folder that contains 'scripts/'."
    )


# =============================================================================
# Memory guard (Mac-safe)
# =============================================================================
def _get_rss_bytes() -> int:
    try:
        import psutil  # type: ignore
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(rss)
        return int(rss) * 1024


def _guard_rss(max_gb: float, note: str = "") -> None:
    rss_gb = _get_rss_bytes() / (1024 ** 3)
    if rss_gb > max_gb:
        raise SystemExit(
            f"[MEMORY GUARD] RSS {rss_gb:.2f} GB exceeds limit {max_gb:.2f} GB. {note}"
        )


# =============================================================================
# Feature extraction helpers (handles forward returning None / dict / tuple)
# =============================================================================
TensorLike = torch.Tensor
OutLike = Union[torch.Tensor, Dict[str, Any], Tuple[Any, ...], None]


def _to_feature_tensor(out: OutLike) -> Optional[torch.Tensor]:
    """
    Heuristics:
    - If tensor: return it
    - If dict: look for common keys
    - If tuple/list: return first tensor found
    - Else: None
    """
    if isinstance(out, torch.Tensor):
        return out

    if isinstance(out, dict):
        # common patterns
        for k in ["feat", "feats", "features", "x", "out", "output", "last_hidden_state"]:
            v = out.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        # fallback: first tensor value
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
        return None

    if isinstance(out, (tuple, list)):
        for v in out:
            if isinstance(v, torch.Tensor):
                return v
            if isinstance(v, dict):
                t = _to_feature_tensor(v)
                if isinstance(t, torch.Tensor):
                    return t
        return None

    return None


@torch.no_grad()
def _infer_feature_dim(backbone: nn.Module, x: torch.Tensor) -> int:
    """
    Try forward(), else forward_features(), then parse output into a tensor.
    Returns last dimension after pooling.
    """
    out = backbone(x)
    t = _to_feature_tensor(out)

    if t is None and hasattr(backbone, "forward_features"):
        out2 = backbone.forward_features(x)  # type: ignore[attr-defined]
        t = _to_feature_tensor(out2)

    if t is None:
        raise RuntimeError(
            "Could not obtain a tensor from backbone. "
            "Your backbone.forward() returned None and forward_features() (if present) "
            "also didn't yield a tensor/dict/tuple containing a tensor."
        )

    # Pool to (B, D)
    emb = _pool_to_embedding(t)
    return int(emb.shape[-1])


def _pool_to_embedding(feat: torch.Tensor) -> torch.Tensor:
    """
    Convert various feature shapes to (B, D) without assuming a specific Swin impl.

    Common cases:
    - (B, D) already
    - (B, S, D) tokens -> mean over S
    - (B, C, H, W) map -> global avg pool
    - (B, C, T, H, W) -> global avg pool
    """
    if feat.ndim == 2:
        return feat

    if feat.ndim == 3:
        # (B, S, D)
        return feat.mean(dim=1)

    if feat.ndim == 4:
        # (B, C, H, W)
        return feat.mean(dim=(2, 3))

    if feat.ndim == 5:
        # (B, C, T, H, W)
        return feat.mean(dim=(2, 3, 4))

    raise RuntimeError(f"Unsupported feature shape for pooling: {tuple(feat.shape)}")


def backbone_forward_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Unified way to get a feature tensor from your backbone even if forward() returns None.
    """
    out = backbone(x)
    t = _to_feature_tensor(out)

    if t is None and hasattr(backbone, "forward_features"):
        out2 = backbone.forward_features(x)  # type: ignore[attr-defined]
        t = _to_feature_tensor(out2)

    if t is None:
        raise RuntimeError(
            "Backbone produced no tensor-like outputs. "
            "Check build_swin2d_backbone() and whether it expects a different call "
            "(e.g., model.forward_features(x) only)."
        )
    return t


# =============================================================================
# Minimal "external SSL architecture" wrapper around a backbone
# =============================================================================
class SimpleSSLArchitecture(nn.Module):
    """
    Mimics:
      wrapper gives backbone
      external architecture uses backbone outputs in SSL
    """

    def __init__(self, backbone: nn.Module, proj_dim: int = 256):
        super().__init__()
        self.backbone = backbone

        # Infer backbone embedding dim once, lazily built via init method
        self._proj_dim = proj_dim
        self.projector: Optional[nn.Module] = None

    def build_projector_if_needed(self, x: torch.Tensor) -> None:
        if self.projector is not None:
            return
        d_in = _infer_feature_dim(self.backbone, x)
        # simple 2-layer MLP projector (BYOL/SimCLR-ish)
        self.projector = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, self._proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized projected embedding z: (B, proj_dim)
        """
        self.build_projector_if_needed(x)
        assert self.projector is not None

        feat = backbone_forward_features(self.backbone, x)
        emb = _pool_to_embedding(feat)               # (B, D)
        z = self.projector(emb)                     # (B, proj_dim)
        z = F.normalize(z, dim=-1)
        return z


def ssl_neg_cosine(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    # maximize cosine similarity => minimize negative cosine
    return -(z1 * z2).sum(dim=-1).mean()


def make_view(x: torch.Tensor, noise_std: float, dropout_p: float) -> torch.Tensor:
    """
    Very lightweight "augmentation" for mel-like tensors:
    - additive gaussian noise
    - element dropout (acts like crude time/freq masking)
    """
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    if dropout_p > 0:
        x = F.dropout(x, p=dropout_p, training=True)
    return x


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    _add_repo_paths_to_syspath()

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--max-rss-gb", type=float, default=6.0)

    # training-ish knobs
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--noise-std", type=float, default=0.01)
    ap.add_argument("--dropout-p", type=float, default=0.05)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--print-param-grads", action="store_true")
    args = ap.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    _guard_rss(args.max_rss_gb, "before audio preprocessing")

    # -------------------------------------------------------------------------
    # Import YOUR audio preprocessor (unchanged)
    # -------------------------------------------------------------------------
    from scripts.preprocessing.audio.AudioPreprocessorNPV import AudioPreprocessorNPV

    preprocessor = AudioPreprocessorNPV()
    mel = preprocessor.process_audio_file(audio_path)

    if not isinstance(mel, torch.Tensor) or mel.ndim != 2:
        raise RuntimeError(
            f"Expected mel Tensor of shape (H, W), got {type(mel)} {getattr(mel, 'shape', None)}"
        )

    # (1,1,H,W) then expand to (B,1,H,W)
    x1 = mel.unsqueeze(0).unsqueeze(0)
    x = x1.expand(args.batch_size, -1, -1, -1).contiguous()

    _guard_rss(args.max_rss_gb, "after mel creation")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available")

    x = x.to(device)

    # -------------------------------------------------------------------------
    # Build Swin Tiny backbone (your builder)
    # -------------------------------------------------------------------------
    from build_swin2d import build_swin2d_backbone

    backbone = build_swin2d_backbone().to(device)
    backbone.train()  # IMPORTANT: training scenario

    # Quick debug: show what forward returns
    with torch.no_grad():
        out_probe = backbone(x[:1])
        if out_probe is None:
            print("[INFO] backbone(x) returned None (expected for some feature-only wrappers).")
        else:
            t_probe = _to_feature_tensor(out_probe)
            print("[INFO] backbone(x) returned:", type(out_probe), "tensor:", None if t_probe is None else tuple(t_probe.shape))

        if hasattr(backbone, "forward_features"):
            out_ff = backbone.forward_features(x[:1])  # type: ignore[attr-defined]
            t_ff = _to_feature_tensor(out_ff)
            print("[INFO] backbone.forward_features(x) returned:",
                  type(out_ff),
                  "tensor:", None if t_ff is None else tuple(t_ff.shape))

    # -------------------------------------------------------------------------
    # External SSL architecture wrapper
    # -------------------------------------------------------------------------
    arch = SimpleSSLArchitecture(backbone=backbone, proj_dim=args.proj_dim).to(device)
    arch.train()

    # Optimizer over *all* params (backbone + projector once created)
    # We'll create projector on first forward, then re-create optimizer safely.
    opt = torch.optim.AdamW(arch.parameters(), lr=args.lr)

    rss_before = _get_rss_bytes() / (1024 ** 3)

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)

        # Two views from the same batch
        v1 = make_view(x, noise_std=0.0, dropout_p=0.0)
        v2 = make_view(x, noise_std=args.noise_std, dropout_p=args.dropout_p)

        z1 = arch(v1)   # (B, proj_dim)
        z2 = arch(v2)   # (B, proj_dim)

        loss = ssl_neg_cosine(z1, z2)
        loss.backward()

        # Check gradient flow into backbone
        with torch.no_grad():
            # pick first parameter that has grad
            grad_norm = None
            for n, p in backbone.named_parameters():
                if p.grad is not None:
                    grad_norm = float(p.grad.detach().norm().cpu())
                    if args.print_param_grads:
                        print(f"  [GRAD] {n}: {grad_norm:.6f}")
                    break

        opt.step()

        print(f"[STEP {step:03d}] loss={float(loss.detach().cpu()):.6f} | backbone_grad_norm={grad_norm}")

        _guard_rss(args.max_rss_gb, f"after step {step}")

    rss_after = _get_rss_bytes() / (1024 ** 3)

    # -------------------------------------------------------------------------
    # Final prints
    # -------------------------------------------------------------------------
    print("\n=== SUMMARY ===")
    print("Audio file:", audio_path)
    print("Mel shape (H, W):", tuple(mel.shape))
    print("Input x (B,C,H,W):", tuple(x.shape))
    print(f"RSS before: {rss_before:.2f} GB | RSS after: {rss_after:.2f} GB")
    print("Done.")


if __name__ == "__main__":
    main()
