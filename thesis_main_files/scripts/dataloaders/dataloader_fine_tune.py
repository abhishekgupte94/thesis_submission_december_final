# dataloader_fine_tune.py
# ============================================================
# [FINAL | STAGE-2 FINE-TUNE DATALOADER]
#
# Agreed design:
#   - Mirror dataloader.py philosophy for filesystem discovery/indexing:
#       * Discover segments from: <offline_root>/<batch_name>/audio/<clip_id>/*.pt
#       * Validate paired files exist:
#           - audio_96:   <clip_id>_<seg_idx>.pt
#           - audio_2048: <clip_id>_<seg_idx>__2048.pt
#           - video:      <video_face_crops>/<clip_id>/seg_<seg_idx>/seg_<seg_idx>.mp4  (FINE-TUNE)
#       * Cache index metadata (paths + T_video) to speed up repeat runs
#   - Remove bucketed batching (standard DataLoader + padding collate)
#   - Keep fine-tune uniqueness: label dictionary (segment-level) loaded from a
#     CSV with columns:
#         clip_id (string), seg_idx (int), label (int 0/1)
#
# Returned batch keys:
#   - audio_96, audio_2048, video_u8_cthw, T_video, y (+ y_onehot in collate)
#
# ============================================================
# [PATCH NOTES | 2025-12-24]
# [MODIFIED] Support video stored as .mp4 (no video tensors available).
# [MODIFIED] IMPORTANT PERFORMANCE FIX:
#   - During indexing, DO NOT decode mp4 for every row.
#   - Use ffprobe to obtain T_video; decode only as fallback if ffprobe fails.
#   - __getitem__ still decodes mp4 normally to return frames.
# ============================================================

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

# ============================================================
# [ADDED][PATCH] Reuse fast ffprobe + mp4 decode helpers from base loader
# (exactly like your other dataloader)
# ============================================================
from scripts.dataloaders.dataloader import (  # noqa: WPS433
    _probe_num_frames_mp4_ffprobe,
    _read_mp4_to_u8_cthw,
)

# ============================================================
# [KEPT] Regex for BASE audio segment files (64x96 only)
# Matches:
#   <clip_id>_<seg_idx>.pt
# Example:
#   1963_0007.pt
#
# Naming contract (STRICT):
#   clip_0007.pt        -> 64x96
#   clip_0007__2048.pt  -> 64x2048
# ============================================================
_AUDIO_RE = re.compile(r"^(?P<clip>.+)_(?P<idx>\d+)\.pt$")


def _resolve_audio_2048_path(audio_96_path: Path) -> Path:
    """Given <clip>_<seg>.pt return <clip>_<seg>__2048.pt"""
    return audio_96_path.with_name(audio_96_path.stem + "__2048" + audio_96_path.suffix)


# ============================================================
# [KEPT] Padding helpers (match base loader philosophy)
# ============================================================
def _pad_mel(x: torch.Tensor, T_max: int) -> torch.Tensor:
    """Pad mel along time axis only. x: (64, T)."""
    if x.shape[1] == T_max:
        return x
    pad = torch.zeros((x.shape[0], T_max - x.shape[1]), dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def _pad_video_u8(v: torch.Tensor, T_max: int) -> torch.Tensor:
    """Pad video along time axis only. v: (3, T, H, W) uint8."""
    if v.shape[1] == T_max:
        return v
    C, T, H, W = v.shape
    pad = torch.zeros((C, T_max - T, H, W), dtype=v.dtype)
    return torch.cat([v, pad], dim=1)


# ============================================================
# [ADDED][PATCH] Video loader supporting BOTH .pt tensors and .mp4 files
#
# - .pt: expects (3,T,H,W) tensor
# - .mp4: decodes to uint8 (3,T,H,W) using base loader helper
#
# Non-breaking contract:
#   returns uint8 (3,T,H,W)
# ============================================================
def _load_video_u8_cthw(video_path: Path) -> torch.Tensor:
    """
    Returns: uint8 (3, T, H, W)
    """
    if video_path.suffix == ".pt":
        v = torch.load(video_path, map_location="cpu")
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"Video .pt did not load a tensor: {video_path}")
        if v.ndim != 4 or v.shape[0] != 3:
            raise ValueError(f"Bad video tensor shape (expected 3xTxHxW): {video_path} got {tuple(v.shape)}")
        return v.to(torch.uint8) if v.dtype != torch.uint8 else v

    if video_path.suffix == ".mp4":
        # [MODIFIED][PATCH] Use imported base helper (no local import)
        v = _read_mp4_to_u8_cthw(video_path)
        if not isinstance(v, torch.Tensor) or v.ndim != 4 or v.shape[0] != 3:
            raise ValueError(f"Bad decoded mp4 tensor shape from: {video_path}")
        return v.to(torch.uint8) if v.dtype != torch.uint8 else v

    raise ValueError(f"Unsupported video suffix: {video_path.suffix} ({video_path})")


# ============================================================
# [KEPT] Labels file loader (segment-level)
#   CSV columns expected:
#       clip_id, seg_idx, label
# ============================================================
def _load_segment_labels_csv(csv_path: Path, *, strict: bool = True) -> Dict[Tuple[str, int], int]:
    labels: Dict[Tuple[str, int], int] = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue
            clip_id = (row.get("clip_id") or "").strip()
            seg_idx_str = (row.get("seg_idx") or "").strip()
            label_str = (row.get("label") or "").strip()
            if not clip_id or not seg_idx_str or not label_str:
                if strict:
                    raise ValueError(f"Bad labels CSV row (missing fields): {row}")
                continue

            try:
                seg_idx = int(seg_idx_str)
                label = int(label_str)
            except Exception:
                if strict:
                    raise ValueError(f"Bad labels CSV row (non-int): {row}")
                continue

            if label not in (0, 1):
                if strict:
                    raise ValueError(f"Bad label value (must be 0/1): {row}")
                continue

            labels[(clip_id, seg_idx)] = label

    return labels


# ============================================================
# Segment Dataset (Stage-2 fine-tune)
# ============================================================
class SegmentDataset(Dataset):
    """Loads (audio_96, audio_2048, video_u8_cthw) per segment."""

    def __init__(
        self,
        *,
        offline_root: Path,
        batch_name: str,
        map_location: str = "cpu",
        strict: bool = False,
        max_segments: Optional[int] = None,
        segments_csv: Optional[Path] = None,
        strict_labels: bool = True,
    ) -> None:
        super().__init__()
        self.offline_root = Path(offline_root)
        self.batch_name = str(batch_name)
        self.batch_dir = self.offline_root / self.batch_name

        self.map_location = str(map_location)
        self.strict = bool(strict)
        self.max_segments = max_segments

        self.audio_root = self.batch_dir / "audio"
        self.video_root = self.batch_dir / "video_face_crops"

        self._segments_csv_path: Optional[Path] = None
        if segments_csv is not None:
            self._segments_csv_path = Path(segments_csv)
        else:
            self._segments_csv_path = self.batch_dir / "segment_index_finetune.csv"

        self.strict_labels = bool(strict_labels)
        self._labels_by_segment: Dict[Tuple[str, int], int] = {}
        if self._segments_csv_path is not None:
            if not self._segments_csv_path.exists():
                if self.strict_labels:
                    raise FileNotFoundError(f"segments_csv not found: {self._segments_csv_path}")
            else:
                self._labels_by_segment = _load_segment_labels_csv(self._segments_csv_path, strict=self.strict_labels)

        # Index tuple:
        #   (clip_id, seg_idx, audio96_path, audio2048_path, video_path, T_video)
        self.index: List[Tuple[str, int, Path, Path, Path, int]] = []
        self._build_index()

    # ============================================================
    # CSV manifest helper
    #
    # Expected CSV header:
    #   clip_id,seg_idx,audio96_rel,audio2048_rel,video_rel,label
    # ============================================================
    def _try_load_segment_index_csv(self, csv_path: Path) -> bool:
        try:
            with csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                required = {"clip_id", "seg_idx", "audio96_rel", "audio2048_rel", "video_rel", "label"}
                if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                    return False

                loaded: List[Tuple[str, int, Path, Path, Path, int]] = []

                for row in reader:
                    if row is None:
                        continue

                    clip_id = (row.get("clip_id") or "").strip()
                    seg_idx_str = (row.get("seg_idx") or "").strip()

                    label_str = (row.get("label") or "").strip()
                    if not clip_id or not seg_idx_str or not label_str:
                        return False
                    try:
                        seg_idx = int(seg_idx_str)
                        label = int(label_str)
                    except Exception:
                        return False
                    if label not in (0, 1):
                        return False

                    a96_rel = (row.get("audio96_rel") or "").strip()
                    a2048_rel = (row.get("audio2048_rel") or "").strip()
                    v_rel = (row.get("video_rel") or "").strip()
                    if not a96_rel or not a2048_rel or not v_rel:
                        return False

                    a96 = self.batch_dir / a96_rel
                    a2048 = self.batch_dir / a2048_rel
                    v_path = self.batch_dir / v_rel

                    if not a96.exists() or not a2048.exists() or not v_path.exists():
                        return False

                    # ------------------------------------------------------------
                    # [MODIFIED][PATCH] capture T_video like the other dataloader:
                    #   - .pt: torch.load to get T
                    #   - .mp4: ffprobe to get T, decode fallback only if probe fails
                    # ------------------------------------------------------------
                    try:
                        if v_path.suffix == ".pt":
                            v = torch.load(v_path, map_location="cpu")
                            if not isinstance(v, torch.Tensor) or v.ndim != 4 or v.shape[0] != 3:
                                return False
                            T_video = int(v.shape[1])

                        elif v_path.suffix == ".mp4":
                            T_video_probe = _probe_num_frames_mp4_ffprobe(v_path)
                            if T_video_probe is None:
                                # fallback: decode once (slow but safe)
                                v_u8 = _read_mp4_to_u8_cthw(v_path)
                                if not isinstance(v_u8, torch.Tensor) or v_u8.ndim != 4 or v_u8.shape[0] != 3:
                                    return False
                                T_video_probe = int(v_u8.shape[1])
                            T_video = int(T_video_probe)

                        else:
                            return False
                    except Exception:
                        return False

                    loaded.append((clip_id, int(seg_idx), a96, a2048, v_path, int(T_video)))

                loaded.sort(key=lambda x: (x[0], x[1]))

                if self.max_segments is not None:
                    loaded = loaded[: self.max_segments]

                self.index = loaded
                return True

        except Exception:
            return False

    # ============================================================
    # Index cache helpers (cache metadata only)
    # ============================================================
    def _index_cache_path(self) -> Path:
        return self.batch_dir / ".segment_index_cache_finetune_v2.json"

    def _index_cache_lock_path(self) -> Path:
        return self.batch_dir / ".segment_index_cache_finetune_v2.lock"

    def _try_load_index_cache(self, cache_path: Path) -> bool:
        try:
            raw = json.loads(cache_path.read_text())
            if not isinstance(raw, list):
                return False

            loaded: List[Tuple[str, int, Path, Path, Path, int]] = []
            for row in raw:
                if not isinstance(row, dict):
                    return False

                clip_id = str(row.get("clip_id"))
                seg_idx = int(row.get("seg_idx"))
                T_video = int(row.get("T_video"))

                a96_rel = row.get("a96_rel")
                a2048_rel = row.get("a2048_rel")
                v_rel = row.get("v_rel")
                if not isinstance(a96_rel, str) or not isinstance(a2048_rel, str) or not isinstance(v_rel, str):
                    return False

                a96 = (self.batch_dir / a96_rel).resolve()
                a2048 = (self.batch_dir / a2048_rel).resolve()
                v_pt = (self.batch_dir / v_rel).resolve()

                if not a96.exists() or not a2048.exists() or not v_pt.exists():
                    return False

                loaded.append((clip_id, seg_idx, a96, a2048, v_pt, T_video))

            if self.max_segments is not None:
                loaded = loaded[: self.max_segments]

            self.index = loaded
            return True
        except Exception:
            return False

    def _save_index_cache(self, cache_path: Path) -> None:
        try:
            payload = []
            for (clip_id, seg_idx, a96, a2048, v_pt, T_video) in self.index:
                payload.append(
                    {
                        "clip_id": clip_id,
                        "seg_idx": int(seg_idx),
                        "T_video": int(T_video),
                        "a96_rel": str(a96.relative_to(self.batch_dir)),
                        "a2048_rel": str(a2048.relative_to(self.batch_dir)),
                        "v_rel": str(v_pt.relative_to(self.batch_dir)),
                    }
                )
            cache_path.write_text(json.dumps(payload))
        except Exception:
            pass

    def _acquire_lock(self, lock_path: Path, timeout_sec: float = 30.0) -> bool:
        t0 = time.time()
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                if (time.time() - t0) > timeout_sec:
                    return False
                time.sleep(0.1)
            except Exception:
                return False

    def _release_lock(self, lock_path: Path) -> None:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    # ============================================================
    # Build index over saved offline segment tensors
    # ============================================================
    def _build_index(self) -> None:
        cache_path = self._index_cache_path()
        lock_path = self._index_cache_lock_path()

        if cache_path.exists():
            if self._try_load_index_cache(cache_path):
                return

        got_lock = self._acquire_lock(lock_path)
        try:
            if cache_path.exists():
                if self._try_load_index_cache(cache_path):
                    return

            # CSV fast-path helper
            if self._segments_csv_path is not None and self._segments_csv_path.exists():
                if self._try_load_segment_index_csv(self._segments_csv_path):
                    self._save_index_cache(cache_path)
                    return

            if not self.audio_root.exists():
                raise FileNotFoundError(f"Missing audio root: {self.audio_root}")
            if not self.video_root.exists():
                raise FileNotFoundError(f"Missing video root: {self.video_root}")

            clip_dirs = [p for p in self.audio_root.iterdir() if p.is_dir()]

            for clip_dir in clip_dirs:
                clip_id = clip_dir.name

                for a96 in clip_dir.glob("*.pt"):
                    m = _AUDIO_RE.match(a96.name)
                    if not m or m.group("clip") != clip_id:
                        continue

                    seg_idx = int(m.group("idx"))

                    a2048 = _resolve_audio_2048_path(a96)
                    if not a2048.exists():
                        if self.strict:
                            raise FileNotFoundError(f"Missing audio_2048: {a2048}")
                        continue

                    # ------------------------------------------------------------
                    # [MODIFIED][PATCH] fine-tune video stored as .mp4 (not .pt)
                    # ------------------------------------------------------------
                    v_path = self.video_root / clip_id / f"seg_{seg_idx:04d}" / f"seg_{seg_idx:04d}.mp4"
                    if not v_path.exists():
                        continue

                    # ------------------------------------------------------------
                    # [MODIFIED][PATCH] Probe T_video without decoding (ffprobe),
                    # fallback decode only if needed
                    # ------------------------------------------------------------
                    T_video_probe = _probe_num_frames_mp4_ffprobe(v_path)
                    if T_video_probe is None:
                        try:
                            v_u8 = _read_mp4_to_u8_cthw(v_path)
                            if not isinstance(v_u8, torch.Tensor) or v_u8.ndim != 4 or v_u8.shape[0] != 3:
                                continue
                            T_video_probe = int(v_u8.shape[1])
                        except Exception:
                            continue

                    T_video = int(T_video_probe)

                    self.index.append((clip_id, seg_idx, a96, a2048, v_path, T_video))

            if self.max_segments is not None:
                self.index = self.index[: self.max_segments]

            self._save_index_cache(cache_path)

        finally:
            if got_lock:
                self._release_lock(lock_path)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, object]:
        clip_id, seg_idx, a96_pt, a2048_pt, v_pt, T_video = self.index[i]

        mel_96 = torch.load(a96_pt, map_location=self.map_location).float()
        mel_2048 = torch.load(a2048_pt, map_location=self.map_location).float()

        # ------------------------------------------------------------
        # [KEPT] decode .mp4 OR load .pt for the actual frames
        # ------------------------------------------------------------
        video = _load_video_u8_cthw(v_pt)

        y = self._labels_by_segment.get((clip_id, int(seg_idx)), None)
        if y is None and self.strict_labels and self._labels_by_segment:
            raise KeyError(f"Missing label for (clip_id={clip_id}, seg_idx={seg_idx}) in segments_csv")

        return {
            "clip_id": clip_id,
            "seg_idx": int(seg_idx),
            "audio": mel_96,  # BACKCOMPAT
            "audio_96": mel_96,
            "audio_2048": mel_2048,
            "video_u8_cthw": video,
            "T_video": int(T_video),
            "y": y,
        }


# ============================================================
# Collate function (padding + masks, no bucketing)
# ============================================================
def collate_segments_pad(items: List[Dict[str, object]]) -> Dict[str, object]:
    clip_ids = [str(it.get("clip_id", "")) for it in items]
    seg_idxs = torch.tensor([int(it.get("seg_idx", -1)) for it in items], dtype=torch.long)

    T_v = max(int(it["T_video"]) for it in items)
    T_a96 = max(int(it["audio_96"].shape[1]) for it in items)
    T_a2048 = max(int(it["audio_2048"].shape[1]) for it in items)

    T_video = torch.tensor([int(it["T_video"]) for it in items], dtype=torch.long)
    T_audio_96 = torch.tensor([int(it["audio_96"].shape[1]) for it in items], dtype=torch.long)
    T_audio_2048 = torch.tensor([int(it["audio_2048"].shape[1]) for it in items], dtype=torch.long)

    videos = torch.stack([_pad_video_u8(it["video_u8_cthw"], T_v) for it in items])
    aud96 = torch.stack([_pad_mel(it["audio_96"], T_a96) for it in items])
    aud2048 = torch.stack([_pad_mel(it["audio_2048"], T_a2048) for it in items])

    # Labels: add y_onehot derivation (B,2) float
    y_list = [it.get("y", None) for it in items]
    if all(y is None for y in y_list):
        y_tensor = None
        y_onehot = None
    else:
        y_tensor = torch.tensor([int(yy) if yy is not None else 0 for yy in y_list], dtype=torch.long)
        y_onehot = torch.nn.functional.one_hot(y_tensor, num_classes=2).float()

    return {
        "clip_ids": clip_ids,
        "seg_idxs": seg_idxs,
        "audio_96": aud96,
        "audio_2048": aud2048,
        "video_u8_cthw": videos,
        "y": y_tensor,
        "y_onehot": y_onehot,
    }


# ============================================================
# Lightning DataModule (no bucketing)
# ============================================================
@dataclass
class SegmentDataModuleFineTuneConfig:
    offline_root: Path
    batch_name: str
    batch_size: int
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    map_location: str = "cpu"
    val_split: float = 0.05
    seed: int = 123
    drop_last: bool = False
    strict: bool = False
    max_segments: Optional[int] = None
    segments_csv: Optional[Path] = None
    strict_labels: bool = True


class SegmentDataModuleFineTune(pl.LightningDataModule):
    def __init__(self, cfg: SegmentDataModuleFineTuneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._train = None
        self._val = None

    def setup(self, stage: Optional[str] = None) -> None:
        ds = SegmentDataset(
            offline_root=self.cfg.offline_root,
            batch_name=self.cfg.batch_name,
            map_location=self.cfg.map_location,
            strict=self.cfg.strict,
            max_segments=self.cfg.max_segments,
            segments_csv=self.cfg.segments_csv,
            strict_labels=self.cfg.strict_labels,
        )

        n = len(ds)
        n_val = max(1, int(round(n * self.cfg.val_split))) if n > 0 else 0
        n_train = max(0, n - n_val)

        g = torch.Generator().manual_seed(int(self.cfg.seed))
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)

        self._train = train_ds
        self._val = val_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            pin_memory=bool(self.cfg.pin_memory),
            persistent_workers=bool(self.cfg.persistent_workers),
            collate_fn=collate_segments_pad,
            drop_last=bool(self.cfg.drop_last),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=int(self.cfg.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.num_workers),
            pin_memory=bool(self.cfg.pin_memory),
            persistent_workers=bool(self.cfg.persistent_workers),
            collate_fn=collate_segments_pad,
            drop_last=False,
        )
