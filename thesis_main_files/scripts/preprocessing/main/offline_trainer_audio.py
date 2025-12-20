#!/usr/bin/env python
"""
offline_export_avspeech_audio_from_video_pt.py

Offline audio export from preprocessed AVSpeech "video_pt" outputs.

Expected input structure
------------------------
<offline_root>/<batch_name>/
- video_pt/<clip_id>_video.pt
- Loads segments_sec from that per-clip video .pt payload
- Calls AudioPreprocessorNPV.process_and_save_from_segments_sec_segmentlocal(...)
  so audio segmentation matches video exactly (no word_time segmentation).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field  # [ADDED] field for default_factory
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from scripts.preprocessing.audio.AudioPreprocessorNPV import (
    AudioPreprocessorNPV,
    AudioPreprocessorNPVConfig,
)  # [MODIFIED] Import cfg dataclass required by AudioPreprocessorNPV.__init__


DEFAULT_OFFLINE_ROOT = "data/processed/AVSpeech/AVSpeech_offline_training_files/audio"
DEFAULT_AUDIO_ROOT = "data/raw/AVSpeech/audio"


def ensure_dirs(cfg: "OfflineAudioExportConfig") -> None:
    cfg.batch_dir.mkdir(parents=True, exist_ok=True)
    cfg.audio_pt_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class OfflineAudioExportConfig:
    audio_root: Path
    offline_root: Path
    batch_name: str

    # ===============================================================
    # [ADDED] Audio preprocessor config
    # Reason: AudioPreprocessorNPV now requires an explicit cfg object.
    # Default behavior remains unchanged because we default to AudioPreprocessorNPVConfig().
    # ===============================================================
    audio_prep_cfg: AudioPreprocessorNPVConfig = field(default_factory=AudioPreprocessorNPVConfig)
    # audio_output_dir: Optional[Path] = None
    @property
    def batch_dir(self) -> Path:
        return self.offline_root / self.batch_name

    @property
    def video_pt_dir(self) -> Path:
        return self.batch_dir / "video_pt"

    @property
    def audio_pt_dir(self) -> Path:
        return self.batch_dir / "audio_pt"

    @property
    def logs_dir(self) -> Path:
        return self.batch_dir / "logs_audio"

    @property
    def index_json(self) -> Path:
        return self.logs_dir / "audio_index.json"


class AudioExportDataset(Dataset[Path]):
    def __init__(self, video_pt_files: Sequence[Path]):
        self.video_pt_files = video_pt_files

    def __len__(self) -> int:
        return len(self.video_pt_files)

    def __getitem__(self, idx: int) -> Path:
        return self.video_pt_files[idx]


class AudioExportDataModule(pl.LightningDataModule):
    def __init__(self, video_pt_files: Sequence[Path], batch_size: int = 1, num_workers: int = 4):
        super().__init__()
        self.video_pt_files = list(video_pt_files)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            AudioExportDataset(self.video_pt_files),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=lambda b: b,
        )


class OfflineAudioExporter(pl.LightningModule):
    def __init__(self, cfg: OfflineAudioExportConfig):
        super().__init__()
        self.cfg = cfg

        self.audio_prep: Optional[AudioPreprocessorNPV] = None

        self.audio_log_csv_rank: Optional[Path] = None
        self.audio_failed_csv_rank: Optional[Path] = None

        self.rank: int = 0
        self.world_size: int = 1

        self.rank_index: Dict[str, Any] = {}
        self.index_json_rank: Optional[Path] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.rank = int(getattr(self.trainer, "global_rank", 0))
        self.world_size = int(getattr(self.trainer, "world_size", 1))

        ensure_dirs(self.cfg)

        self.audio_log_csv_rank = self.cfg.logs_dir / f"audio_export_log_rank{self.rank}.csv"
        self.audio_failed_csv_rank = self.cfg.logs_dir / f"audio_export_failed_rank{self.rank}.csv"
        self.index_json_rank = self.cfg.logs_dir / f"audio_index_rank{self.rank}.json"

        if self.audio_prep is None:
            self.audio_prep = AudioPreprocessorNPV(cfg=self.cfg.audio_prep_cfg)  # [MODIFIED] pass cfg (no behavior change with defaults)

    def _process_one_video_pt(self, video_pt_path: Path) -> Dict[str, Any]:
        assert self.audio_prep is not None
        assert self.audio_log_csv_rank is not None

        video_payload = torch.load(video_pt_path, map_location="cpu")

        clip_id = video_payload.get("clip_id", None)
        if clip_id is None:
            clip_id = video_pt_path.stem.replace("_video", "")

        segments_sec = video_payload.get("segments_sec", None)
        if segments_sec is None:
            raise RuntimeError(f"Missing 'segments_sec' in {video_pt_path}")

        audio_input_path = self.cfg.audio_root / f"{clip_id}.wav"
        if not audio_input_path.exists():
            audio_input_path = self.cfg.audio_root / f"{clip_id}.mp3"

        out = self.audio_prep.process_and_save_from_segments_sec_segmentlocal(
            audio_path=audio_input_path,
            clip_id=clip_id,
            segments_sec=segments_sec,
            out_pt_path=self.cfg.audio_pt_dir
        )
        num_segments, num_words = out
        return {
            "ok": True,
            "clip_id": clip_id,
            "rank": self.rank,
            "video_pt": str(video_pt_path),
            "audio_path": str(audio_input_path),
            "n_segments": int(num_segments),
        }

    def predict_step(self, batch: List[Path], batch_idx: int, dataloader_idx: int = 0):
        assert self.audio_log_csv_rank is not None
        assert self.audio_failed_csv_rank is not None

        rows_ok: List[Dict[str, Any]] = []
        rows_fail: List[Dict[str, Any]] = []

        for video_pt_path in batch:
            try:
                r = self._process_one_video_pt(video_pt_path)
                rows_ok.append(r)
            except Exception as e:
                rows_fail.append(
                    {
                        "ok": False,
                        "rank": self.rank,
                        "video_pt": str(video_pt_path),
                        "error": repr(e),
                    }
                )

        if rows_ok:
            write_header = not self.audio_log_csv_rank.exists()
            with self.audio_log_csv_rank.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_ok[0].keys()))
                if write_header:
                    w.writeheader()
                for r in rows_ok:
                    w.writerow(r)

        if rows_fail:
            write_header = not self.audio_failed_csv_rank.exists()
            with self.audio_failed_csv_rank.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_fail[0].keys()))
                if write_header:
                    w.writeheader()
                for r in rows_fail:
                    w.writerow(r)

        return {"ok": len(rows_ok), "fail": len(rows_fail)}

    def on_predict_end(self) -> None:
        if self.index_json_rank is None:
            return

        payload = {
            "rank": self.rank,
            "world_size": self.world_size,
            "audio_log_csv_rank": str(self.audio_log_csv_rank) if self.audio_log_csv_rank else None,
            "audio_failed_csv_rank": str(self.audio_failed_csv_rank) if self.audio_failed_csv_rank else None,
        }
        with self.index_json_rank.open("w") as f:
            json.dump(payload, f, indent=2)


def list_video_pt_files(video_pt_dir: Path) -> List[Path]:
    if not video_pt_dir.exists():
        raise FileNotFoundError(f"video_pt_dir does not exist: {video_pt_dir}")
    return sorted(video_pt_dir.glob("*_video.pt"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=str, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--offline-root", type=str, default=DEFAULT_OFFLINE_ROOT)
    parser.add_argument("--batch-name", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    args = parser.parse_args()

    cfg = OfflineAudioExportConfig(
        audio_root=Path(args.audio_root),
        offline_root=Path(args.offline_root),
        batch_name=args.batch_name,
    )
    ensure_dirs(cfg)

    video_pt_files = list_video_pt_files(cfg.video_pt_dir)

    dm = AudioExportDataModule(video_pt_files, batch_size=args.batch_size, num_workers=args.num_workers)
    model = OfflineAudioExporter(cfg)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=8,
        strategy="ddp",
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.predict(model, datamodule=dm)


if __name__ == "__main__":
    main()
