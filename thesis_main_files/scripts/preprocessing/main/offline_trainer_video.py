#!/usr/bin/env python
"""
offline_trainer_video.py

Stage-1 VIDEO exporter (Lightning DDP), JSON timestamps version.

This script:
- Loads timestamps from a single JSON: { "<video_id>": [[start_sec,end_sec], ...], ... }
- Resolves each video file under:  <video_root>/<video_id>/...
- Calls VideoPreprocessorNPV.process_and_save_facecrops_to_disk_from_word_times_segmentlocal(...)
  (Your VideoPreprocessorNPV now saves ONE .mp4 per segment instead of frames.)

IMPORTANT PATCH (MINIMUM, DDP/GPU UNCHANGED):
--------------------------------------------
PyTorch Lightning reserves attributes like `local_rank` (read-only property).
Setting `self.local_rank = ...` inside LightningModule raises:
  AttributeError: can't set attribute 'local_rank'

So we store ranks in non-colliding attributes:
  self._global_rank, self._local_rank, self._world_size

GPU binding semantics remain IDENTICAL:
  ctx_id = self._local_rank
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# NOTE: This import assumes your repo root is on PYTHONPATH (or you run with -m from repo root).
from scripts.preprocessing.video.VideoPreprocessorNPV import VideoPreprocessorNPV, VideoPreprocessorConfig


DEFAULT_OFFLINE_ROOT = "data/processed/AVSpeech/AVSpeech_offline_training_files"


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
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

    @property
    def video_pt_dir(self) -> Path:
        return self.batch_dir / "video_pt"

    @property
    def logs_dir(self) -> Path:
        return self.batch_dir / "logs"

    @property
    def index_json(self) -> Path:
        return self.batch_dir / "video_index.json"


def ensure_dirs(cfg: OfflineVideoExportConfig) -> None:
    cfg.batch_dir.mkdir(parents=True, exist_ok=True)
    cfg.crops_out_dir.mkdir(parents=True, exist_ok=True)
    cfg.video_pt_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)


def _rel_to_offline_root(path: Path, offline_root: Path) -> str:
    try:
        return path.resolve().relative_to(offline_root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


# ======================================================================
# Resolve video path from directory video_root/<video_id>/
# ======================================================================
def resolve_video_path(video_root: Path, video_id: str) -> Path:
    """
    Resolve a concrete video file path given:
      - video_root/<video_id>/... (directory)
    Strategy:
      1) If video_root/<video_id> is a file -> use it.
      2) If it's a directory:
         - try <video_id>.mp4/.mkv/.avi/.mov inside it
         - else pick the first video-like file by extension
    """
    base = video_root / video_id

    if base.exists() and base.is_file():
        return base

    if base.exists() and base.is_dir():
        for ext in [".mp4", ".mkv", ".avi", ".mov"]:
            cand = base / f"{video_id}{ext}"
            if cand.exists():
                return cand

        for ext in [".mp4", ".mkv", ".avi", ".mov"]:
            hits = sorted(base.glob(f"*{ext}"))
            if hits:
                return hits[0]

        return base / f"{video_id}.mp4"

    return video_root / f"{video_id}.mp4"


# ======================================================================
# JSON loader: { video_id: [[s,e],...], ... }
# ======================================================================
def load_timestamps_json(path: Path) -> Dict[str, List[List[float]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("timestamps_json must be a dict: {video_id: [[s,e],...], ...}")

    out: Dict[str, List[List[float]]] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            k = str(k)
        if not isinstance(v, list):
            continue

        word_times: List[List[float]] = []
        for pair in v:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                s = float(pair[0])
                e = float(pair[1])
            except Exception:
                continue
            if e > s:
                word_times.append([s, e])

        word_times.sort(key=lambda x: x[0])
        out[k] = word_times

    return out


# ----------------------------------------------------------------------
# Dataset / DataModule
# ----------------------------------------------------------------------
class VideoExportDataset(Dataset):
    def __init__(self, items: List[Tuple[str, List[List[float]]]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[str, List[List[float]]]:
        return self.items[idx]


class VideoExportDataModule(pl.LightningDataModule):
    def __init__(self, items: List[Tuple[str, List[List[float]]]], batch_size: int = 1, num_workers: int = 2):
        super().__init__()
        self.items = items
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            VideoExportDataset(self.items),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=lambda b: b,  # list[(video_id, word_times), ...]
        )


# ----------------------------------------------------------------------
# LightningModule
# ----------------------------------------------------------------------
class OfflineVideoExporter(pl.LightningModule):
    def __init__(self, cfg: OfflineVideoExportConfig):
        super().__init__()
        self.cfg = cfg

        # ================================================================
        # [MODIFIED] Avoid Lightning attribute collisions.
        # Lightning reserves `local_rank` (read-only). Setting it breaks.
        # Store ranks in our own non-colliding attributes.
        # ================================================================
        self._global_rank: int = 0
        self._local_rank: int = 0
        self._world_size: int = 1

        self.video_prep: Optional[VideoPreprocessorNPV] = None

        self.rank_index: Dict[str, Any] = {}
        self.index_json_rank: Optional[Path] = None
        self.video_log_csv_rank: Optional[Path] = None

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
        # ================================================================
        # [MODIFIED] Store ranks in non-colliding attributes
        # ================================================================
        self._global_rank = int(getattr(self.trainer, "global_rank", 0))
        self._world_size = int(getattr(self.trainer, "world_size", 1))
        self._local_rank = int(getattr(self.trainer, "local_rank", 0))

        ensure_dirs(self.cfg)

        self.index_json_rank = self.cfg.logs_dir / f"video_index_rank{self._global_rank}.json"
        self.video_log_csv_rank = self.cfg.logs_dir / f"video_export_log_rank{self._global_rank}.csv"

        # ================================================================
        # [CRITICAL] DDP-safe InsightFace GPU binding (UNCHANGED SEMANTICS):
        # each process uses its own GPU via ctx_id=local_rank
        # ================================================================
        if self.video_prep is None:
            vp_cfg = VideoPreprocessorConfig(
                ctx_id=self._local_rank,  # [MODIFIED] was self.local_rank
                use_gpu_if_available=True,
            )
            self.video_prep = VideoPreprocessorNPV(cfg=vp_cfg)

    def _process_one(self, video_id: str, word_times: List[List[float]]) -> Dict[str, Any]:
        assert self.video_prep is not None
        assert self.video_log_csv_rank is not None

        t0 = time.time()

        video_path = resolve_video_path(self.cfg.video_root, video_id)
        info: Dict[str, Any] = {
            "video_id": video_id,
            "rank": self._global_rank,       # [MODIFIED] was self.rank
            "local_rank": self._local_rank,  # [MODIFIED] was self.local_rank
            "video_path": str(video_path),
            "timestamps_json": str(self.cfg.timestamps_json),
        }

        if not word_times:
            info["status"] = "error"
            info["error"] = "empty_word_times"
            info["proc_time_sec"] = time.time() - t0
            return info

        if not video_path.exists():
            info["status"] = "error"
            info["error"] = "missing_video"
            info["proc_time_sec"] = time.time() - t0
            return info

        # Outputs
        video_pt_path = self.cfg.video_pt_dir / f"{video_id}_video.pt"
        crops_out_dir = self.cfg.crops_out_dir

        # Resume-safety
        if video_pt_path.exists():
            info["status"] = "skipped"
            info["video_pt"] = str(video_pt_path)
            info["proc_time_sec"] = time.time() - t0
            return info

        try:
            # ================================================================
            # Your VideoPreprocessorNPV now saves ONE .mp4 per segment.
            # Function signature stays the same, so trainer stays DDP/GPU-identical.
            #
            # keep_full_when_no_face=False -> only true face crops are saved.
            # out_pt_path -> ensures segments_sec etc. are saved into the .pt payload.
            # ================================================================
            num_segments, total_saved = self.video_prep.process_and_save_facecrops_to_disk_from_word_times_segmentlocal(
                video_path=video_path,
                word_times=word_times,
                out_dir=crops_out_dir,
                keep_full_when_no_face=False,
                out_pt_path=video_pt_path,
            )

            info.update(
                dict(
                    status="ok",
                    video_pt=str(video_pt_path),
                    num_segments=int(num_segments),
                    saved_frames=int(total_saved),  # kept for backwards compatibility
                    saved_segment_videos=int(num_segments),  # one mp4 per segment
                    crops_format="segment_mp4",
                    proc_time_sec=time.time() - t0,
                )
            )

            self._append_row(
                self.video_log_csv_rank,
                header=[
                    "video_id",
                    "video_file",
                    "video_pt",
                    "num_segments",
                    "saved_frames",
                    "saved_segment_videos",
                    "crops_format",
                    "rank",
                    "proc_time_sec",
                ],
                row=[
                    video_id,
                    video_path.name,
                    str(video_pt_path),
                    int(num_segments),
                    int(total_saved),
                    int(num_segments),
                    "segment_mp4",
                    self._global_rank,  # [MODIFIED] was self.rank
                    info["proc_time_sec"],
                ],
            )

            return info

        except Exception as e:
            info["status"] = "error"
            info["error"] = str(e)
            info["proc_time_sec"] = time.time() - t0
            return info

    def predict_step(self, batch, batch_idx: int):
        for video_id, word_times in batch:
            info = self._process_one(video_id=video_id, word_times=word_times)
            self.rank_index[video_id] = info
        return None

    def on_predict_end(self) -> None:
        assert self.index_json_rank is not None

        with self.index_json_rank.open("w", encoding="utf-8") as f:
            json.dump(self.rank_index, f, indent=2)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._global_rank == 0:  # [MODIFIED] was self.rank
            merged: Dict[str, Any] = {}
            for r in range(self._world_size):  # [MODIFIED] was self.world_size
                p = self.cfg.logs_dir / f"video_index_rank{r}.json"
                if p.exists():
                    with p.open("r", encoding="utf-8") as f:
                        merged.update(json.load(f))

            # portable paths
            for vid, info in merged.items():
                if "video_pt" in info:
                    info["video_pt"] = _rel_to_offline_root(Path(info["video_pt"]), self.cfg.offline_root)
                if "video_path" in info:
                    info["video_path"] = _rel_to_offline_root(Path(info["video_path"]), self.cfg.offline_root)
                if "timestamps_json" in info:
                    info["timestamps_json"] = _rel_to_offline_root(Path(info["timestamps_json"]), self.cfg.offline_root)

            with self.cfg.index_json.open("w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)

            print(f"[DONE] Video index written to: {self.cfg.index_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 VIDEO exporter from timestamps JSON (Lightning DDP).")

    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--timestamps-json", type=str, required=True)

    parser.add_argument("--offline-root", type=str, default=DEFAULT_OFFLINE_ROOT)
    parser.add_argument("--batch-name", type=str, required=True)

    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")

    args = parser.parse_args()

    cfg = OfflineVideoExportConfig(
        video_root=Path(args.video_root),
        timestamps_json=Path(args.timestamps_json),
        offline_root=Path(args.offline_root),
        batch_name=args.batch_name,
    )
    ensure_dirs(cfg)

    if not cfg.timestamps_json.exists():
        raise FileNotFoundError(f"timestamps_json not found: {cfg.timestamps_json}")

    ts_map = load_timestamps_json(cfg.timestamps_json)

    if not ts_map:
        raise ValueError(f"No usable timestamps found in: {cfg.timestamps_json}")

    items: List[Tuple[str, List[List[float]]]] = sorted(ts_map.items(), key=lambda kv: kv[0])
    print(f"[INFO] Loaded {len(items)} entries from timestamps JSON.")

    dm = VideoExportDataModule(items=items, batch_size=args.batch_size, num_workers=args.num_workers)
    module = OfflineVideoExporter(cfg)

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
