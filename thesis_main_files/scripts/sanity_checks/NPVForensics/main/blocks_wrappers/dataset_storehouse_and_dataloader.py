#!/usr/bin/env python
"""
Sanity check for:
- AVSegmentTokensDataset / AVSegmentTokenDataModule
- AVTokenizerSegmentsDataset / AVTokenizerDataModule

Creates a temporary mini dataset on disk and loads it.
"""

import json
import os
import tempfile
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader

from scripts.dataloaders.dataset_storehouse  import AVSegmentTokensDataset, AVTokenizerSegmentsDataset
from  scripts.dataloaders.dataloader import AVSegmentTokenDataModule, AVTokenizerDataModule

# [ADDED]
from utils.memory_guard.memory_guard import MemoryGuard


def make_dummy_segment_tokens_dataset(root: str, guard: MemoryGuard) -> str:
    """
    Create a tiny AVSegmentTokensDataset index + .pt files.

    Returns the path to the index.json.
    """
    guard.check()  # [ADDED] before making dirs/allocations

    os.makedirs(os.path.join(root, "tokens"), exist_ok=True)

    entries: List[Dict[str, Any]] = []
    for i in range(3):
        guard.check()  # [ADDED] inside loop

        sample_id = f"video{i:03d}_seg000"
        tokens_path = os.path.join("tokens", f"{sample_id}.pt")

        audio_tokens = torch.randn(5, 32)  # [Sa, D_a]
        video_tokens = torch.randn(7, 64)  # [Sv, D_v]
        torch.save({"audio_tokens": audio_tokens, "video_tokens": video_tokens},
                   os.path.join(root, tokens_path))

        entries.append(
            {
                "id": sample_id,
                "tokens_path": tokens_path,
                "video_id": f"video{i:03d}",
                "segment_idx": 0,
                "split": "train" if i < 2 else "val",
                "label": i % 2,
            }
        )

    index_path = os.path.join(root, "index_segment_tokens.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    return index_path


def make_dummy_tokenizer_dataset(root: str, guard: MemoryGuard) -> str:
    """
    Create a tiny AVTokenizerSegmentsDataset index + .pt files.

    Returns the path to the index.json.
    """
    guard.check()  # [ADDED]

    os.makedirs(os.path.join(root, "audio"), exist_ok=True)
    os.makedirs(os.path.join(root, "video"), exist_ok=True)

    entries: List[Dict[str, Any]] = []
    for i in range(2):
        guard.check()  # [ADDED] inside loop

        clip_id = f"clip_{i:04d}"
        audio_pt_rel = os.path.join("audio", f"{clip_id}_audio.pt")
        video_pt_rel = os.path.join("video", f"{clip_id}_video.pt")

        # Audio payload
        mel_segments = [torch.randn(80, 64) for _ in range(3)]  # 3 segments
        audio_payload = {
            "mel_segments": mel_segments,
            "segments_sec": [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
        }
        torch.save(audio_payload, os.path.join(root, audio_pt_rel))

        # Video payload
        video_segments = [torch.randn(16, 3, 112, 112) for _ in range(3)]
        video_payload = {
            "video_segments": video_segments,
            "segments_sec": [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
        }
        torch.save(video_payload, os.path.join(root, video_pt_rel))

        entries.append(
            {
                "id": clip_id,
                "audio_pt": audio_pt_rel,
                "video_pt": video_pt_rel,
                "timestamps_csv": None,
                "split": "train" if i == 0 else "val",
                "label": i % 2,
            }
        )

    index_path = os.path.join(root, "index_tokenizer.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    return index_path


def main():
    # [ADDED] Guards
    guard_strict = MemoryGuard(max_process_gb=8.0, min_system_available_gb=2.0, throws=True)
    guard_soft = MemoryGuard(max_process_gb=8.0, min_system_available_gb=2.0, throws=False)

    guard_strict.check()  # [ADDED]

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Temp root:", tmpdir)

        guard_soft.check()  # [ADDED]

        # ---- Segment tokens dataset + DM ----
        seg_index = make_dummy_segment_tokens_dataset(tmpdir, guard_soft)
        seg_dm = AVSegmentTokenDataModule(
            index_json_path=seg_index,
            batch_size=2,
            num_workers=0,   # safe for Mac; on A100 you can increase this
            root_dir=tmpdir,
            drop_last=False,
        )

        seg_dm.setup("fit")
        guard_soft.check()  # [ADDED]

        batch = next(iter(seg_dm.train_dataloader()))
        print("\n[SegmentTokenDataModule] batch keys:", batch.keys())
        print(" audio_tokens:", batch["audio_tokens"].shape)
        print(" video_tokens:", batch["video_tokens"].shape)

        guard_soft.check()  # [ADDED]

        # ---- Tokenizer dataset + DM ----
        tok_index = make_dummy_tokenizer_dataset(tmpdir, guard_soft)
        tok_dm = AVTokenizerDataModule(
            index_json_path=tok_index,
            batch_size=2,
            num_workers=0,
            root_dir=tmpdir,
            drop_last=False,
        )
        tok_dm.setup("fit")

        guard_soft.check()  # [ADDED]

        tbatch = next(iter(tok_dm.train_dataloader()))
        print("\n[AVTokenizerDataModule] batch keys:", tbatch.keys())
        print(" mel_stack:", tbatch["mel_stack"].shape)
        print(" #segment_tensors:", len(tbatch["segment_tensors"]))

        guard_soft.check()  # [ADDED]


if __name__ == "__main__":
    main()
