#!/usr/bin/env python
"""
offline_trainer_video.py

Stage-1 VIDEO exporter (Lightning DDP).

IMPORTANT:
- This file is IDENTICAL to your original.
- The ONLY change is a drop-in replacement of VideoPreprocessorNPV import.
- No logic, GPU handling, DDP behavior, paths, saving, or trainer flow is modified.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# [ONLY CHANGE IN THIS FILE]
# Drop-in replacement of the VideoPreprocessor implementation.
# NOTHING else in this file is modified.
# ============================================================================
# OLD (original):
# from scripts.preprocessing.video.VideoPreprocessorNPV import (
#     VideoPreprocessorNPV,
#     VideoPreprocessorConfig,
# )

# NEW (timestamp-driven, same API, same behavior, extra .pt saved per segment):
from scripts.preprocessing.video.VideoPreprocessorNPV_TimestampDriven import (
    VideoPreprocessorNPV,
    VideoPreprocessorConfig,
)
# ============================================================================


# ============================================================================
# Config
# ============================================================================
@dataclass
class OfflineVideoExportConfig:
    video_root: Path
    timestamps_json: Path
    offline_root: Path
    batch_name: str

    @property
    def batch_dir(self) -> Path:
        return self.offline_root / self.batch_name

    @property
    def crops_out_dir(self) -> Path:
        return self.batch_dir / "video_face_crops"


# ============================================================================
# Dataset
# ============================================================================
class TimestampJSONVideoDataset(Dataset):
    """
    timestamps_json format:
      {
        "<video_id>": [[start_sec, end_sec], [start_sec, end_sec], ...],
        ...
      }
    """

    def __init__(self, video_root: Path, timestamps_json: Path):
        self.video_root = Path(video_root)
        self.timestamps_json = Path(timestamps_json)

        with self.timestamps_json.open("r", encoding="utf-8") as f:
            self.db: Dict[str, Any] = json.load(f)

        self.video_ids = sorted(self.db.keys())

    def __len__(self) -> int:
        return len(self.video_ids)

    def _resolve_video_path(self, video_id: str) -> Path:
        # Same resolution logic as original
        p1 = self.video_root / f"{video_id}.mp4"
        if p1.exists():
            return p1

        p2 = self.video_root / video_id / f"{video_id}.mp4"
        if p2.exists():
            return p2

        pdir = self.video_root / video_id
        if pdir.exists() and pdir.is_dir():
            files = sorted(p for p in pdir.iterdir() if p.is_file())
            if files:
                return files[0]

        raise FileNotFoundError(
            f"Could not resolve video path for video_id={video_id} under {self.video_root}"
        )

    def __getitem__(self, idx: int) -> Tuple[str, Path, List[List[float]]]:
        video_id = self.video_ids[idx]
        segments = self.db[video_id]
        video_path = self._resolve_video_path(video_id)
        return video_id, video_path, segments


def collate_identity(batch):
    # Unchanged: batch items are passed through as-is
    return batch


# ============================================================================
# Lightning exporter
# ============================================================================
class OfflineVideoExporter(pl.LightningModule):
    def __init__(self, cfg: OfflineVideoExportConfig):
        super().__init__()
        self.cfg = cfg

        # IMPORTANT: Lightning owns local_rank; do not override it
        self.rank_id = int(os.environ.get("LOCAL_RANK", "0"))

        # VideoPreprocessor configuration unchanged
        vp_cfg = VideoPreprocessorConfig(
            ctx_id=self.rank_id,
        )
        self.vp = VideoPreprocessorNPV(vp_cfg)

    def configure_optimizers(self):
        # No optimizer (export-only job)
        return None

    def training_step(self, batch, batch_idx: int):
        for (video_id, video_path, segments) in batch:
            self.vp.process_and_save_facecrops_to_disk_from_word_times_segmentlocal(
                video_path=video_path,
                word_times=segments,
                out_dir=self.cfg.crops_out_dir,
                keep_full_when_no_face=False,
            )

        # Dummy tensor to satisfy Lightning
        return torch.tensor(0.0, device=self.device)


# ============================================================================
# Main
# ============================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True, type=str)
    ap.add_argument("--timestamps-json", required=True, type=str)
    ap.add_argument("--offline-root", required=True, type=str)
    ap.add_argument("--batch-name", required=True, type=str)
    ap.add_argument("--devices", required=True, type=int)
    ap.add_argument("--num-workers", default=4, type=int)
    ap.add_argument("--precision", default=16, type=int)
    return ap.parse_args()


def main():
    args = parse_args()

    cfg = OfflineVideoExportConfig(
        video_root=Path(args.video_root),
        timestamps_json=Path(args.timestamps_json),
        offline_root=Path(args.offline_root),
        batch_name=args.batch_name,
    )

    ds = TimestampJSONVideoDataset(cfg.video_root, cfg.timestamps_json)
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_identity,
        pin_memory=True,
    )

    exporter = OfflineVideoExporter(cfg)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=int(args.devices),
        strategy="ddp" if int(args.devices) > 1 else "auto",
        max_epochs=1,
        precision=int(args.precision),
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    trainer.fit(exporter, train_dataloaders=dl)


if __name__ == "__main__":
    main()
