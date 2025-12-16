#!/usr/bin/env python
"""
offline_export_avspeech_audio_from_video_pt.py

Stage-2 AUDIO exporter (Lightning DDP).
- Enumerates Stage-1 video_pt/<clip_id>_video.pt
- Loads segments_sec from that per-clip video .pt payload
- Calls AudioPreprocessorNPV.process_and_save_from_segments_sec_segmentlocal(...)
  so audio segmentation matches video exactly (no word_time segmentation).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from scripts.preprocessing.audio.AudioPreprocessorNPV import AudioPreprocessorNPV


DEFAULT_OFFLINE_ROOT = "data/processed/AVSpeech/AVSpeech_offline_training_files/audio"


@dataclass
class OfflineAudioExportConfig:
    audio_root: Path
    offline_root: Path
    batch_name: str

    @property
    def batch_dir(self) -> Path:
        return self.offline_root / self.batch_name

    @property
    def video_pt_dir(self) -> Path:
        return self.batch_dir / "video_pt"

    @property
    def audio_pt_dir(self) -> Path:
        return self.batch_dir / "audio"

    @property
    def logs_dir(self) -> Path:
        return self.batch_dir / "logs"

    @property
    def index_json(self) -> Path:
        return self.batch_dir / "audio_index.json"


def ensure_dirs(cfg: OfflineAudioExportConfig) -> None:
    cfg.batch_dir.mkdir(parents=True, exist_ok=True)
    cfg.video_pt_dir.mkdir(parents=True, exist_ok=True)
    cfg.audio_pt_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)


def _rel_to_offline_root(path: Path, offline_root: Path) -> str:
    try:
        return path.resolve().relative_to(offline_root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


class AudioExportDataset(Dataset):
    def __init__(self, video_pt_files: List[Path]):
        self.video_pt_files = video_pt_files

    def __len__(self) -> int:
        return len(self.video_pt_files)

    def __getitem__(self, idx: int) -> str:
        return str(self.video_pt_files[idx])


class AudioExportDataModule(pl.LightningDataModule):
    def __init__(self, video_pt_files: List[Path], batch_size: int = 1, num_workers: int = 4):
        super().__init__()
        self.video_pt_files = video_pt_files
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
        self.rank: int = 0
        self.world_size: int = 1

        self.rank_index: Dict[str, Any] = {}
        self.index_json_rank: Optional[Path] = None
        self.audio_log_csv_rank: Optional[Path] = None

    @staticmethod
    def _append_row(csv_path: Path, header: List[str], row: List[Any]) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerow(row)

    def setup(self, stage: Optional[str] = None) -> None:
        self.rank = int(getattr(self.trainer, "global_rank", 0))
        self.world_size = int(getattr(self.trainer, "world_size", 1))

        ensure_dirs(self.cfg)

        self.index_json_rank = self.cfg.logs_dir / f"audio_index_rank{self.rank}.json"
        self.audio_log_csv_rank = self.cfg.logs_dir / f"audio_export_log_rank{self.rank}.csv"

        if self.audio_prep is None:
            self.audio_prep = AudioPreprocessorNPV()

    def _process_one_video_pt(self, video_pt_path: Path) -> Dict[str, Any]:
        assert self.audio_prep is not None
        assert self.audio_log_csv_rank is not None

        t0 = time.time()

        clip_id = video_pt_path.name.replace("_video.pt", "")
        audio_path = self.cfg.audio_root / f"{clip_id}.wav"

        # ============================================================
        # [PATCH] audio_pt_path is now an OUTPUT ROOT DIRECTORY
        # ============================================================
        audio_pt_path = self.cfg.audio_pt_dir

        info: Dict[str, Any] = {
            "clip_id": clip_id,
            "rank": self.rank,
            "video_pt": str(video_pt_path),
            "audio_path": str(audio_path),
        }

        if not audio_path.exists():
            info["status"] = "error"
            info["error"] = "missing_audio"
            info["proc_time_sec"] = time.time() - t0
            return info

        if not video_pt_path.exists():
            info["status"] = "error"
            info["error"] = "missing_video_pt"
            info["proc_time_sec"] = time.time() - t0
            return info

        # ============================================================
        # [PATCH] Skip check is now per-clip directory
        # ============================================================
        clip_audio_dir = self.cfg.audio_pt_dir / clip_id
        if clip_audio_dir.exists():
            info["status"] = "skipped"
            info["audio_pt"] = str(clip_audio_dir)
            info["proc_time_sec"] = time.time() - t0
            return info

        # ------------------------------------------------------------------
        # Load segments_sec from Stage-1 video .pt
        # ------------------------------------------------------------------
        vp = torch.load(video_pt_path, map_location="cpu")
        segments_sec = vp.get("segments_sec", None)
        if not segments_sec:
            info["status"] = "error"
            info["error"] = "missing_segments_sec_in_video_pt"
            info["proc_time_sec"] = time.time() - t0
            return info

        try:
            # ------------------------------------------------------------------
            # [PATCH] segment-driven audio export (per-segment .pt)
            # ------------------------------------------------------------------
            num_segments, _ = self.audio_prep.process_and_save_from_segments_sec_segmentlocal(
                audio_path=audio_path,
                segments_sec=segments_sec,
                out_pt_path=audio_pt_path,     # directory
                log_csv_path=self.audio_log_csv_rank,
                clip_id=clip_id,               # required for naming
            )

            info.update(
                dict(
                    status="ok",
                    audio_pt=str(clip_audio_dir),
                    num_segments=int(num_segments),
                    proc_time_sec=time.time() - t0,
                )
            )

            self._append_row(
                csv_path=self.audio_log_csv_rank,
                header=["clip_id", "audio_file", "audio_pt", "num_segments", "rank", "proc_time_sec"],
                row=[clip_id, audio_path.name, str(clip_audio_dir), num_segments, self.rank, info["proc_time_sec"]],
            )

            return info

        except Exception as e:
            info["status"] = "error"
            info["error"] = str(e)
            info["proc_time_sec"] = time.time() - t0
            return info

    def predict_step(self, batch, batch_idx: int):
        for video_pt_str in batch:
            info = self._process_one_video_pt(Path(video_pt_str))
            self.rank_index[info["clip_id"]] = info
        return None

    def on_predict_end(self) -> None:
        assert self.index_json_rank is not None

        with self.index_json_rank.open("w", encoding="utf-8") as f:
            json.dump(self.rank_index, f, indent=2)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.rank == 0:
            merged: Dict[str, Any] = {}
            for r in range(self.world_size):
                p = self.cfg.logs_dir / f"audio_index_rank{r}.json"
                if p.exists():
                    with p.open("r", encoding="utf-8") as f:
                        merged.update(json.load(f))

            for clip_id, info in merged.items():
                if "video_pt" in info:
                    info["video_pt"] = _rel_to_offline_root(Path(info["video_pt"]), self.cfg.offline_root)
                if "audio_pt" in info:
                    info["audio_pt"] = _rel_to_offline_root(Path(info["audio_pt"]), self.cfg.offline_root)
                if "audio_path" in info:
                    info["audio_path"] = _rel_to_offline_root(Path(info["audio_path"]), self.cfg.offline_root)

            with self.cfg.index_json.open("w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)

            print(f"[DONE] Audio index written to: {self.cfg.index_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 AUDIO exporter from Stage-1 video .pt segments (Lightning DDP).")
    parser.add_argument("--audio-root", type=str, required=True)
    parser.add_argument("--offline-root", type=str, default=DEFAULT_OFFLINE_ROOT)
    parser.add_argument("--batch-name", type=str, required=True)

    parser.add_argument("--devices", type=int, default=8)
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

    video_pt_files = sorted(cfg.video_pt_dir.glob("*_video.pt"))
    if not video_pt_files:
        raise FileNotFoundError(f"No Stage-1 video pt files found under: {cfg.video_pt_dir}")

    print(f"[INFO] Found {len(video_pt_files)} Stage-1 video .pt files.")

    dm = AudioExportDataModule(
        video_pt_files=video_pt_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    module = OfflineAudioExporter(cfg)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp",
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.predict(module, datamodule=dm)


if __name__ == "__main__":
    main()
