#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import torch

# If you put the dataloader code in another file, import it here:
from scripts.dataloaders.dataloader import ClipAsBatchDataModule

# ---- If you didn't split files, paste your dataloader code ABOVE this line ----


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", type=str, required=True, help="Root <batch_dir> containing audio/ and video_face_crops/")
    ap.add_argument("--clips-csv", type=str, required=True, help="CSV containing clip_ids")
    ap.add_argument("--clip-id-column", type=str, default=None, help="Optional CSV header name for clip_id")
    ap.add_argument("--num-workers", type=int, default=0, help="Use 0 first for debugging on Mac")
    ap.add_argument("--max-batches", type=int, default=3)
    args = ap.parse_args()

    dm = ClipAsBatchDataModule(
        batch_dir=args.batch_dir,
        clips_csv=args.clips_csv,
        clip_id_column=args.clip_id_column,
        num_workers=args.num_workers,
        pin_memory=False,              # Mac CPU test; keep simple
        persistent_workers=False,
    )

    dm.setup("fit")
    dl = dm.train_dataloader()

    print("\n=== Sanity check: iterating dataloader ===")
    for bi, batch in enumerate(dl):
        clip_id = batch["clip_id"]
        audio = batch["audio"]                 # (S,64,96)
        video_paths = batch["video_paths"]     # list of Paths length S

        # batch_size=1 collate returns clip_id as a scalar string, not list
        print(f"\n[BATCH {bi}] clip_id = {clip_id}")
        print(f"  audio dtype={audio.dtype} device={audio.device} shape={tuple(audio.shape)}")
        print(f"  video_paths type={type(video_paths)} len={len(video_paths)}")

        # Basic invariants
        S = audio.shape[0]
        assert len(video_paths) == S, f"Sa != Sv: Sa={S} Sv={len(video_paths)}"

        # Check audio shape
        assert audio.ndim == 3 and audio.shape[1:] == (64, 96), f"Bad audio shape: {tuple(audio.shape)}"

        # Check that the segment indices appear consistent by filename
        # (Optional: just a helpful check)
        first = Path(video_paths[0]).name
        last = Path(video_paths[-1]).name
        print(f"  first video file: {first}")
        print(f"  last  video file: {last}")

        # Decode 1 video to confirm decode works (optional / cheap)
        # You can comment this out if you don't have decord/torchvision video support.
        try:
            v0 = _decode_video_cthw_all_frames(Path(video_paths[0]))
            print(f"  decoded video[0] shape (C,T,H,W) = {tuple(v0.shape)}  dtype={v0.dtype}")
        except Exception as e:
            print(f"  (video decode skipped/failed) reason: {e}")

        if bi + 1 >= args.max_batches:
            break

    print("\n✅ Sanity check passed (basic alignment + shapes).")


if __name__ == "__main__":
    main()
