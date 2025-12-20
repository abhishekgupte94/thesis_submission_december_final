# dataloader_fine_tune.py
# ============================================================
# [FINAL | STAGE-2 FINE-TUNE DATALOADER]
#
# Purpose:
#   - Segment-based fine-tuning dataloader
#   - Loads paired audio + video tensors saved offline
#
# Audio (per segment):
#   - audio_96   : (64, 96)   short-context log-Mel
#   - audio_2048 : (64, 2048) long-context log-Mel
#   - audio      : alias for audio_96 (BACKCOMPAT)
#
# Video (per segment):
#   - video_u8_cthw : (3, T, H, W), uint8
#   - video         : alias (BACKCOMPAT)
#
# Labels:
#   - y        : (B,) int or None
#   - y_onehot : (B,2) or None
#
# Guarantees:
#   - NO rewiring of sampler logic
#   - NO change to CSV handling semantics
#   - NO assumptions about model usage
#   - Explicit Stage-2 alignment
# ============================================================

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import lightning as pl


# ============================================================
# [KEPT] Regex for BASE audio segment files (64x96 only)
# Matches:
#   <clip_id>_<seg_idx:04d>.pt
# ============================================================
_AUDIO_RE = re.compile(r"^(?P<clip>.+)_(?P<idx>\d{4})\.pt$")


# ============================================================
# [ADDED | STAGE-2]
# Resolve paired 64x2048 audio tensor
#
# Naming contract (STRICT):
#   clip_0007.pt        -> 64x96
#   clip_0007__2048.pt  -> 64x2048
#
# Rationale:
#   - Avoids accidental glob collisions
#   - Keeps 96-frame files authoritative
# ============================================================
def _resolve_audio_2048_path(audio_96_path: Path) -> Path:
    return audio_96_path.with_name(audio_96_path.stem + "__2048" + audio_96_path.suffix)


# ============================================================
# [KEPT] CSV helpers (unchanged semantics)
# ============================================================
def _read_csv_manifest(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _infer_clip_id_from_row(row: Dict[str, str]) -> Optional[str]:
    for k in ("clip_id", "clip", "id"):
        if k in row and row[k].strip():
            return row[k].strip()
    for k in ("file", "filename", "path"):
        if k in row and row[k].strip():
            return Path(row[k].strip()).stem
    return None


def _infer_label_from_row(row: Dict[str, str]) -> Optional[int]:
    for k in ("label", "y", "class", "target"):
        if k in row and row[k].strip():
            return int(float(row[k]))
    return None


def _one_hot_2(y: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((y.numel(), 2), dtype=torch.float32, device=y.device)
    out.scatter_(1, y.view(-1, 1), 1.0)
    return out


# ============================================================
# Dataset
# ============================================================
class SegmentDataset(Dataset):
    """
    [STAGE-2 DATASET]

    Each sample returns:
      - audio_96      : (64,96)
      - audio_2048    : (64,2048)
      - audio         : alias of audio_96
      - video_u8_cthw : (3,T,H,W), uint8
      - T_video       : int
      - y             : int or None
    """

    def __init__(
        self,
        *,
        batch_dir: Path,
        map_location: str = "cpu",
        strict: bool = True,
        csv_manifest: Optional[Path] = None,
        strict_csv: bool = False,
        max_segments: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.batch_dir = Path(batch_dir)
        self.map_location = map_location
        self.strict = strict
        self.strict_csv = strict_csv
        self.max_segments = max_segments

        # ------------------------------------------------------------
        # [KEPT] Directory layout
        # ------------------------------------------------------------
        self.audio_root = self.batch_dir / "audio"
        self.video_root = self.batch_dir / "video_face_crops"

        # ------------------------------------------------------------
        # [KEPT] CSV filtering + labels
        # ------------------------------------------------------------
        self._labels_by_clip_id: Dict[str, int] = {}
        self._clip_ids_filter: Optional[List[str]] = None

        if csv_manifest is not None:
            rows = _read_csv_manifest(csv_manifest)
            clip_ids: List[str] = []
            for r in rows:
                cid = _infer_clip_id_from_row(r)
                if cid:
                    clip_ids.append(cid)
                    lab = _infer_label_from_row(r)
                    if lab is not None:
                        self._labels_by_clip_id[cid] = lab
            self._clip_ids_filter = sorted(set(clip_ids))

        # ------------------------------------------------------------
        # Index tuple:
        #   (clip_id, seg_idx, audio96, audio2048, video, T_video)
        # ------------------------------------------------------------
        self.index: List[Tuple[str, int, Path, Path, Path, int]] = []
        self._build_index()

    # ============================================================
    # [MODIFIED | STAGE-2]
    # Build index using ONLY base audio_96 files
    # ============================================================
    def _build_index(self) -> None:
        clip_dirs = (
            [self.audio_root / cid for cid in self._clip_ids_filter]
            if self._clip_ids_filter
            else [p for p in self.audio_root.iterdir() if p.is_dir()]
        )

        for clip_dir in clip_dirs:
            if not clip_dir.exists():
                continue

            clip_id = clip_dir.name

            for a96 in clip_dir.glob("*.pt"):
                m = _AUDIO_RE.match(a96.name)
                if not m:
                    continue

                seg_idx = int(m.group("idx"))

                # [ADDED] paired long-context audio
                a2048 = _resolve_audio_2048_path(a96)
                if not a2048.exists():
                    if self.strict:
                        raise FileNotFoundError(f"Missing audio_2048: {a2048}")
                    continue

                v_pt = self.video_root / clip_id / f"seg_{seg_idx:04d}" / f"seg_{seg_idx:04d}.pt"
                if not v_pt.exists():
                    continue

                v = torch.load(v_pt, map_location="cpu")
                if not isinstance(v, torch.Tensor) or v.ndim != 4 or v.shape[0] != 3:
                    continue

                self.index.append(
                    (clip_id, seg_idx, a96, a2048, v_pt, int(v.shape[1]))
                )

        if self.max_segments is not None:
            self.index = self.index[: self.max_segments]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, object]:
        clip_id, seg_idx, a96_pt, a2048_pt, v_pt, T_video = self.index[i]

        # ------------------------------------------------------------
        # [ADDED] Load audio tensors
        # ------------------------------------------------------------
        mel_96 = torch.load(a96_pt, map_location=self.map_location).float()
        mel_2048 = torch.load(a2048_pt, map_location=self.map_location).float()

        # ------------------------------------------------------------
        # [CRITICAL] Load + enforce uint8 video
        # ------------------------------------------------------------
        video = torch.load(v_pt, map_location="cpu")
        if video.dtype != torch.uint8:
            video = video.to(torch.uint8)

        # ------------------------------------------------------------
        # [KEPT] Label handling
        # ------------------------------------------------------------
        y = self._labels_by_clip_id.get(clip_id, None)

        return {
            "audio": mel_96,            # BACKCOMPAT
            "audio_96": mel_96,         # STAGE-2
            "audio_2048": mel_2048,     # STAGE-2
            "video_u8_cthw": video,     # CRITICAL
            "T_video": T_video,
            "y": y,
        }


# ============================================================
# Padding helpers
# ============================================================
def _pad_video_u8(v: torch.Tensor, T: int) -> torch.Tensor:
    if v.shape[1] >= T:
        return v[:, :T]
    return torch.cat([v, v[:, -1:].repeat(1, T - v.shape[1], 1, 1)], dim=1)


def _pad_mel(m: torch.Tensor, T: int) -> torch.Tensor:
    if m.shape[1] >= T:
        return m[:, :T]
    return torch.cat([m, m[:, -1:].repeat(1, T - m.shape[1])], dim=1)


# ============================================================
# [STAGE-2] Collate function
# ============================================================
def collate_segments_bucket_pad(items: List[Dict[str, object]]) -> Dict[str, object]:
    T_v = max(it["T_video"] for it in items)
    T_a96 = max(it["audio_96"].shape[1] for it in items)
    T_a2048 = max(it["audio_2048"].shape[1] for it in items)

    videos = torch.stack([_pad_video_u8(it["video_u8_cthw"], T_v) for it in items])
    aud96 = torch.stack([_pad_mel(it["audio_96"], T_a96) for it in items])
    aud2048 = torch.stack([_pad_mel(it["audio_2048"], T_a2048) for it in items])

    y_list = [it["y"] for it in items]
    if all(y is None for y in y_list):
        y_tensor = None
        y_onehot = None
    else:
        y_tensor = torch.tensor([y if y is not None else 0 for y in y_list])
        y_onehot = _one_hot_2(y_tensor)

    return {
        "audio": aud96,               # BACKCOMPAT
        "audio_96": aud96,
        "audio_2048": aud2048,
        "video_u8_cthw": videos,      # CRITICAL
        "video": videos,              # ALIAS
        "y": y_tensor,
        "y_onehot": y_onehot,
    }


# ============================================================
# Lightning DataModule
# ============================================================
class SegmentDataModuleFineTune(pl.LightningDataModule):
    def __init__(
        self,
        *,
        offline_root: Path,
        batch_name: str,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        map_location: str = "cpu",
        val_split: float = 0.05,
        seed: int = 123,
        bucket_size: int = 8,
        drop_last: bool = True,
        strict: bool = True,
        csv_manifest: Optional[Path] = None,
        strict_csv: bool = False,
        max_segments: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.offline_root = Path(offline_root)
        self.batch_name = batch_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.map_location = map_location
        self.val_split = val_split
        self.seed = seed
        self.bucket_size = bucket_size
        self.drop_last = drop_last
        self.strict = strict
        self.csv_manifest = csv_manifest
        self.strict_csv = strict_csv
        self.max_segments = max_segments

        self._train = None
        self._val = None

    def setup(self, stage: Optional[str] = None) -> None:
        batch_dir = self.offline_root / self.batch_name

        ds = SegmentDataset(
            batch_dir=batch_dir,
            map_location=self.map_location,
            strict=self.strict,
            csv_manifest=self.csv_manifest,
            strict_csv=self.strict_csv,
            max_segments=self.max_segments,
        )

        n = len(ds)
        n_val = max(1, int(round(n * self.val_split)))

        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).tolist()

        val_idx = set(perm[:n_val])
        train_idx = [i for i in perm if i not in val_idx]

        self._train = torch.utils.data.Subset(ds, train_idx)
        self._val = torch.utils.data.Subset(ds, list(val_idx))

        if train_idx:
            self.batch_size = min(self.batch_size, len(train_idx))

    def train_dataloader(self) -> DataLoader:
        lengths = [self._train.dataset.index[i][5] for i in self._train.indices]

        sampler = BucketBatchSampler(
            lengths=lengths,
            batch_size=self.batch_size,
            bucket_size=self.bucket_size,
            shuffle=True,
            drop_last=self.drop_last,
            seed=self.seed,
        )

        return DataLoader(
            self._train,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_segments_bucket_pad,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_segments_bucket_pad,
            drop_last=False,
        )


# ============================================================
# Bucket sampler (unchanged)
# ============================================================
class BucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        *,
        lengths: Sequence[int],
        batch_size: int,
        bucket_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        self.lengths = list(map(int, lengths))
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self._buckets: Dict[int, List[int]] = {}
        for idx, L in enumerate(self.lengths):
            key = (L // self.bucket_size) * self.bucket_size
            self._buckets.setdefault(key, []).append(idx)

        self._keys = sorted(self._buckets.keys())

    def __iter__(self) -> Iterable[List[int]]:
        g = torch.Generator().manual_seed(self.seed)
        keys = self._keys

        if self.shuffle:
            perm = torch.randperm(len(keys), generator=g).tolist()
            keys = [keys[i] for i in perm]

        for k in keys:
            idxs = self._buckets[k]
            if self.shuffle:
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]

            batch: List[int] = []
            for ix in idxs:
                batch.append(ix)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if batch and not self.drop_last:
                yield batch

    def __len__(self) -> int:
        n = 0
        for idxs in self._buckets.values():
            q, r = divmod(len(idxs), self.batch_size)
            n += q
            if r and not self.drop_last:
                n += 1
        return n