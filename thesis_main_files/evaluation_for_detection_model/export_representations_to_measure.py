# evaluation_for_detection_model/export_intermediates.py
from __future__ import annotations

"""
============================================================
export_intermediates.py

ROLE (SSL PROBING)
-----------------
Extracts internal representations WITHOUT modifying
the architecture.

Technique:
- Forward hooks
- DDP-safe
- Non-invasive

This is how serious SSL systems are evaluated.
============================================================
"""

import argparse
from pathlib import Path
import torch

from dataloader import SegmentDataModule
from evaluation_for_detection_model.build_model import build_model, load_weights_into_model, BuildModelArgs
from evaluation_for_detection_model.utils_dist import rank


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offline-root", required=True)
    p.add_argument("--batch-name", required=True)
    p.add_argument("--weights-pt", default="checkpoints/best_weights.pt")
    p.add_argument("--swin2d-ckpt", required=True)
    p.add_argument("--swin3d-ckpt", required=True)
    p.add_argument("--out-dir", default="ssl_representations")
    p.add_argument("--limit-batches", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dm = SegmentDataModule(
        offline_root=Path(args.offline_root),
        batch_name=args.batch_name,
        batch_size=4,
        num_workers=4,
        val_split=0.05,
    )
    dm.setup(stage="fit")
    loader = dm.val_dataloader()

    model = build_model(BuildModelArgs(
        swin2d_ckpt=args.swin2d_ckpt,
        swin3d_ckpt=args.swin3d_ckpt,
        device=device,
    ))
    load_weights_into_model(model, args.weights_pt)

    out_dir = Path(args.out_dir)
    if rank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    # === Hook target: Pre-VACL token unifier ===
    def hook_fn(_, __, output):
        captured["tokens"] = output

    h = model.pre_vacl_unifier.register_forward_hook(hook_fn)

    for bi, batch in enumerate(loader):
        if bi >= args.limit_batches:
            break

        audio = batch["audio"].to(device).unsqueeze(1)
        video = batch["video_u8_cthw"].to(device).float().div_(255.0)
        _ = model(video_in=video, audio_in=audio)

        if "tokens" in captured and rank() == 0:
            torch.save({
                "clip_id": batch["clip_id"],
                "seg_idx": batch["seg_idx"],
                "tokens": captured["tokens"],
            }, out_dir / f"batch_{bi:05d}.pt")

    h.remove()
    if rank() == 0:
        print(f"[EXPORTED] SSL representations → {out_dir}")


if __name__ == "__main__":
    main()
