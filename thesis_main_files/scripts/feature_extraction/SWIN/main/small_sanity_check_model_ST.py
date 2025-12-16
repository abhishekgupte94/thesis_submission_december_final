#!/usr/bin/env python
"""
sanity_audio_preprocessor_swin_tiny_noresize.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


# =============================================================================
# [ADDED] Make imports work no matter where you run this script from
# Reason: 'scripts....' lives under your repo root (or under thesis_main_files),
# but Python only auto-adds the directory containing THIS file.
# So we push the correct root(s) into sys.path at runtime.
# =============================================================================
def _add_repo_paths_to_syspath() -> None:
    this_file = Path(__file__).resolve()

    # Option A: if sanity is inside thesis_main_files/**, then thesis_main_files is an ancestor
    for p in [this_file.parent, *this_file.parents]:
        if p.name == "thesis_main_files":
            repo_root = p.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))        # so "scripts...." works
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))                # so local modules also work
            return

    # Option B: fallback (if you put sanity somewhere else):
    # go up a few levels and hope we hit the repo root that contains "scripts/"
    for p in [this_file.parent, *this_file.parents]:
        if (p / "scripts").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return

    # If neither worked, show a clear error
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
# Main sanity
# =============================================================================
def main() -> None:
    # [ADDED] must happen before importing your 'scripts....' module
    _add_repo_paths_to_syspath()

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--max-rss-gb", type=float, default=6.0)
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

    x = mel.unsqueeze(0).unsqueeze(0)  # (1, 1, 96, 64) (no resize)

    _guard_rss(args.max_rss_gb, "after mel creation")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available")

    x = x.to(device)

    # -------------------------------------------------------------------------
    # Build Swin Tiny backbone (patched config already applied)
    # -------------------------------------------------------------------------
    from build_swin2d import build_swin2d_backbone

    model = build_swin2d_backbone().to(device).eval()

    rss_before = _get_rss_bytes() / (1024 ** 3)

    with torch.no_grad():
        out = model(x)

    rss_after = _get_rss_bytes() / (1024 ** 3)
    _guard_rss(args.max_rss_gb, "after Swin forward")

    print("Audio file:", audio_path)
    print("Mel shape (H, W):", tuple(mel.shape))
    print("Input to Swin (B,C,H,W):", tuple(x.shape))
    print("Swin output shape:", tuple(out.shape))
    print(f"RSS before: {rss_before:.2f} GB | RSS after: {rss_after:.2f} GB")


if __name__ == "__main__":
    main()
