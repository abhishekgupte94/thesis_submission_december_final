#!/usr/bin/env python
"""
offline_export_avspeech.py

Stage-2 offline exporter for AVSpeech-style data.

[MODIFIED] This version is:
- PyTorch Lightning driven using the *predict* loop (no training).
- DDP-safe on an 8×A100 box:
    • Each rank gets a disjoint subset (DistributedSampler).
    • Each rank writes rank-local logs + partial index.
    • Rank0 merges partial indices into a single av_index.json.
- Updated to YOUR pasted preprocessors:
    • CSV is parsed HERE -> word_times = [[start_sec, end_sec], ...]
    • AudioPreprocessorNPV expects word_times + out_pt_path (+ optional log_csv_path)
    • VideoPreprocessorNPV expects word_times + out_pt_path (NO log_csv_path argument)
- [CRITICAL] InsightFace GPU binding is fixed:
    • VideoPreprocessorConfig(ctx_id=local_rank) so each DDP process uses its own GPU.
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
from torch.utils.data import DataLoader, Dataset

from audio_preprocessor_npv import AudioPreprocessorNPV
from VideoPreprocessorNPV import VideoPreprocessorNPV, VideoPreprocessorConfig


# ----------------------------------------------------------------------
# Default root for offline training artifacts
# ----------------------------------------------------------------------
DEFAULT_OFFLINE_ROOT = "data/processed/AVSpeech/AVSpeech_offline_training_files"


# ----------------------------------------------------------------------
# Config for a single offline export run/batch
# ----------------------------------------------------------------------
@dataclass
class OfflineExportConfig:
    audio_root: Path
    video_root: Path
    timestamps_root: Path
    offline_root: Path
    batch_name: str

    @property
    def batch_dir(self) -> Path:
        return self.offline_root / self.batch_name

    @property
    def audio_out_dir(self) -> Path:
        return self.batch_dir / "audio"

    @property
    def video_out_dir(self) -> Path:
        return self.batch_dir / "video"

    @property
    def logs_dir(self) -> Path:
        return self.batch_dir / "logs"

    @property
    def index_json(self) -> Path:
        return self.batch_dir / "av_index.json"


# ----------------------------------------------------------------------
# Directory setup
# ----------------------------------------------------------------------
def ensure_dirs(cfg: OfflineExportConfig) -> None:
    cfg.batch_dir.mkdir(parents=True, exist_ok=True)
    cfg.audio_out_dir.mkdir(parents=True, exist_ok=True)
    cfg.video_out_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Discover *_words.csv timestamp files
# ----------------------------------------------------------------------
def find_timestamps_files(timestamps_root: Path) -> Dict[str, Path]:
    """Return mapping clip_id -> ts_csv_path (for *_words.csv)."""
    mapping: Dict[str, Path] = {}
    for csv_path in sorted(timestamps_root.glob("*_words.csv")):
        stem = csv_path.stem
        if not stem.endswith("_words"):
            continue
        clip_id = stem[:-6]  # remove "_words"
        mapping[clip_id] = csv_path
    return mapping


# ----------------------------------------------------------------------
# Guess video path from clip_id
# ----------------------------------------------------------------------
def guess_video_path(video_root: Path, clip_id: str) -> Path:
    cand = video_root / f"{clip_id}.mp4"
    if cand.exists():
        return cand
    for ext in [".mkv", ".avi", ".mov"]:
        alt = video_root / f"{clip_id}{ext}"
        if alt.exists():
            return alt
    return cand


# ----------------------------------------------------------------------
# Path normalisation for av_index.json
# ----------------------------------------------------------------------
def _rel_to_offline_root(path: Path, offline_root: Path) -> str:
    """Return POSIX path relative to offline_root where possible."""
    try:
        rel = path.resolve().relative_to(offline_root.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


# ======================================================================
# [ADDED] CSV -> in-memory word_times loader (matches your preprocessors)
# ======================================================================
def load_word_times_from_words_csv(ts_csv: Path) -> List[List[float]]:
    """
    Parse Whisper-style *_words.csv with columns "start" and "end" into:
        [[start_sec, end_sec], ...]
    """
    word_times: List[List[float]] = []
    with ts_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = float(row["start"])
                e = float(row["end"])
            except Exception:
                continue
            if e > s:
                word_times.append([s, e])

    word_times.sort(key=lambda x: x[0])
    return word_times


# ======================================================================
# [ADDED] Dataset + DataModule so Lightning can shard with DDP cleanly
# ======================================================================
class OfflineExportDataset(Dataset):
    def __init__(self, items: List[Tuple[str, Path]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        clip_id, ts_csv = self.items[idx]
        return clip_id, str(ts_csv)


class OfflineExportDataModule(pl.LightningDataModule):
    def __init__(self, items: List[Tuple[str, Path]], batch_size: int = 1, num_workers: int = 4):
        super().__init__()
        self.items = items
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def predict_dataloader(self) -> DataLoader:
        ds = OfflineExportDataset(self.items)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=lambda batch: batch,  # list[(clip_id, ts_csv_str), ...]
        )


# ======================================================================
# [ADDED] LightningModule that performs offline export in predict_step()
# ======================================================================
class OfflineExporterModule(pl.LightningModule):
    """
    DDP-safe offline exporter.

    - Each rank processes only its own sampler subset.
    - Each rank writes:
        logs/audio_export_log_rank{rank}.csv
        logs/video_export_log_rank{rank}.csv
        logs/av_index_rank{rank}.json
    - Rank0 merges -> av_index.json
    """

    def __init__(self, cfg: OfflineExportConfig):
        super().__init__()
        self.cfg = cfg

        # created in setup() per-process
        self.audio_prep: Optional[AudioPreprocessorNPV] = None
        self.video_prep: Optional[VideoPreprocessorNPV] = None

        self.rank: int = 0
        self.local_rank: int = 0
        self.world_size: int = 1

        self.rank_index: Dict[str, Any] = {}

        self.audio_log_csv_rank: Optional[Path] = None
        self.video_log_csv_rank: Optional[Path] = None
        self.index_json_rank: Optional[Path] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.rank = int(getattr(self.trainer, "global_rank", 0))
        self.world_size = int(getattr(self.trainer, "world_size", 1))
        self.local_rank = int(getattr(self.trainer, "local_rank", 0))

        ensure_dirs(self.cfg)

        # ------------------------------------------------------------------
        # [MODIFIED] DDP-safe logs/index: one file per rank
        # ------------------------------------------------------------------
        self.audio_log_csv_rank = self.cfg.logs_dir / f"audio_export_log_rank{self.rank}.csv"
        self.video_log_csv_rank = self.cfg.logs_dir / f"video_export_log_rank{self.rank}.csv"
        self.index_json_rank = self.cfg.logs_dir / f"av_index_rank{self.rank}.json"

        # ------------------------------------------------------------------
        # [MODIFIED] Preprocessors instantiated per-process.
        # [CRITICAL] InsightFace must bind to the correct GPU per process.
        #           Your VideoPreprocessorConfig defaults ctx_id=0, so without
        #           this, all ranks would hammer GPU0 -> contention/OOM.
        # ------------------------------------------------------------------
        if self.audio_prep is None:
            self.audio_prep = AudioPreprocessorNPV()

        if self.video_prep is None:
            vp_cfg = VideoPreprocessorConfig(
                ctx_id=self.local_rank,          # <- DDP-safe GPU binding
                use_gpu_if_available=True,
            )
            self.video_prep = VideoPreprocessorNPV(cfg=vp_cfg)

    # ------------------------------------------------------------------
    # [ADDED] tiny helper: append one row to a CSV (rank-local)
    # ------------------------------------------------------------------
    @staticmethod
    def _append_row(csv_path: Path, header: List[str], row: List[Any]) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerow(row)

    def _process_one_clip(self, clip_id: str, ts_csv: Path) -> Dict[str, Any]:
        assert self.audio_prep is not None
        assert self.video_prep is not None
        assert self.audio_log_csv_rank is not None
        assert self.video_log_csv_rank is not None

        audio_path = self.cfg.audio_root / f"{clip_id}.wav"
        video_path = guess_video_path(self.cfg.video_root, clip_id)

        t0 = time.time()

        info: Dict[str, Any] = {
            "clip_id": clip_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "audio_path": str(audio_path),
            "video_path": str(video_path),
            "timestamps_csv": str(ts_csv),
        }

        if not audio_path.exists():
            info["status"] = "error"
            info["error"] = "missing_audio"
            info["proc_time_sec"] = time.time() - t0
            return info

        if not video_path.exists():
            info["status"] = "error"
            info["error"] = "missing_video"
            info["proc_time_sec"] = time.time() - t0
            return info

        audio_pt_path = self.cfg.audio_out_dir / f"{clip_id}_audio.pt"
        video_pt_path = self.cfg.video_out_dir / f"{clip_id}_video.pt"

        # Resume safety
        if audio_pt_path.exists() and video_pt_path.exists():
            info["status"] = "skipped"
            info["audio_pt"] = str(audio_pt_path)
            info["video_pt"] = str(video_pt_path)
            info["proc_time_sec"] = time.time() - t0
            return info

        # ------------------------------------------------------------------
        # [MODIFIED] CSV -> in-memory word_times
        # ------------------------------------------------------------------
        word_times = load_word_times_from_words_csv(ts_csv)
        if not word_times:
            info["status"] = "error"
            info["error"] = "empty_word_times"
            info["proc_time_sec"] = time.time() - t0
            return info

        # AUDIO
        num_segments_a = 0
        if not audio_pt_path.exists():
            try:
                num_segments_a, num_words_a = self.audio_prep.process_and_save_from_timestamps_csv_segmentlocal(
                    audio_path=audio_path,
                    word_times=word_times,
                    out_pt_path=audio_pt_path,
                    log_csv_path=self.audio_log_csv_rank,  # audio supports this
                )
            except Exception as e:
                info["status"] = "error"
                info["audio_error"] = str(e)
                info["proc_time_sec"] = time.time() - t0
                return info
        else:
            num_words_a = len(word_times)

        # VIDEO
        num_segments_v = 0
        if not video_pt_path.exists():
            try:
                # NOTE: your pasted VideoPreprocessorNPV does NOT accept log_csv_path
                num_segments_v, num_words_v = self.video_prep.process_and_save_from_timestamps_csv_segmentlocal(
                    video_path=video_path,
                    word_times=word_times,
                    out_pt_path=video_pt_path,
                    keep_full_when_no_face=True,
                )
            except Exception as e:
                info["status"] = "error"
                info["video_error"] = str(e)
                info["proc_time_sec"] = time.time() - t0
                return info
        else:
            num_words_v = len(word_times)

        # ------------------------------------------------------------------
        # [ADDED] rank-local video log (since video preprocessor has no log arg)
        # ------------------------------------------------------------------
        self._append_row(
            csv_path=self.video_log_csv_rank,
            header=["clip_id", "video_file", "video_pt", "num_words", "num_segments", "rank", "proc_time_sec"],
            row=[
                clip_id,
                video_path.name,
                str(video_pt_path),
                int(num_words_v),
                int(num_segments_v),
                int(self.rank),
                float(time.time() - t0),
            ],
        )

        info.update(
            dict(
                status="ok",
                audio_pt=str(audio_pt_path),
                video_pt=str(video_pt_path),
                num_words_audio=int(num_words_a),
                num_words_video=int(num_words_v),
                num_segments_audio=int(num_segments_a),
                num_segments_video=int(num_segments_v),
                proc_time_sec=time.time() - t0,
            )
        )
        return info

    def predict_step(self, batch, batch_idx: int):
        for clip_id, ts_csv_str in batch:
            ts_csv = Path(ts_csv_str)
            info = self._process_one_clip(clip_id=clip_id, ts_csv=ts_csv)
            self.rank_index[clip_id] = info
        return None

    def on_predict_end(self) -> None:
        assert self.index_json_rank is not None

        # Write partial index for this rank
        with self.index_json_rank.open("w", encoding="utf-8") as f:
            json.dump(self.rank_index, f, indent=2)

        # Barrier before merge
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Rank0 merges
        if self.rank == 0:
            merged: Dict[str, Any] = {}

            # Merge rank partials
            for r in range(self.world_size):
                rank_file = self.cfg.logs_dir / f"av_index_rank{r}.json"
                if not rank_file.exists():
                    continue
                with rank_file.open("r", encoding="utf-8") as f:
                    part = json.load(f)

                for clip_id, info in part.items():
                    # Make paths portable (relative to offline_root where possible)
                    if "audio_pt" in info:
                        info["audio_pt"] = _rel_to_offline_root(Path(info["audio_pt"]), self.cfg.offline_root)
                    if "video_pt" in info:
                        info["video_pt"] = _rel_to_offline_root(Path(info["video_pt"]), self.cfg.offline_root)
                    if "timestamps_csv" in info:
                        info["timestamps_csv"] = _rel_to_offline_root(Path(info["timestamps_csv"]), self.cfg.offline_root)
                    merged[clip_id] = info

            # Write final merged index
            self.cfg.index_json.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.index_json.open("w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)

            print(f"[DONE] Merged index written to: {self.cfg.index_json}")


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Offline AVSpeech exporter (Lightning DDP).")

    parser.add_argument("--audio-root", type=str, required=True)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--timestamps-root", type=str, required=True)

    parser.add_argument("--offline-root", type=str, default=DEFAULT_OFFLINE_ROOT)
    parser.add_argument("--batch-name", type=str, required=True)

    # DDP knobs
    parser.add_argument("--devices", type=int, default=8, help="GPUs to use (8 for 8×A100 box).")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per rank.")
    parser.add_argument("--batch-size", type=int, default=1, help="Clips per rank per step (keep 1 if heavy).")
    parser.add_argument("--precision", type=str, default="32", help="Keep 32 for safety (offline export).")

    args = parser.parse_args()

    cfg = OfflineExportConfig(
        audio_root=Path(args.audio_root),
        video_root=Path(args.video_root),
        timestamps_root=Path(args.timestamps_root),
        offline_root=Path(args.offline_root),
        batch_name=args.batch_name,
    )

    ensure_dirs(cfg)

    # Discover timestamps
    ts_mapping = find_timestamps_files(cfg.timestamps_root)
    if not ts_mapping:
        print(f"[ERROR] No *_words.csv files found under: {cfg.timestamps_root}")
        return

    items: List[Tuple[str, Path]] = sorted(ts_mapping.items(), key=lambda kv: kv[0])
    print(f"[INFO] Found {len(items)} timestamp files total.")

    dm = OfflineExportDataModule(items=items, batch_size=args.batch_size, num_workers=args.num_workers)
    module = OfflineExporterModule(cfg=cfg)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp",
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # Using predict loop for distributed offline work (no training)
    trainer.predict(module, datamodule=dm)


if __name__ == "__main__":
    main()
