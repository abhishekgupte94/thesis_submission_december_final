# dataloader.py
# ============================================================
# [PATCHED] Segment DataModule with:
#   - Train/Val split (val_split)
#   - train/val dataloaders
#   - Bucketing sampler for TRAIN only
#   - Simple sequential batches for VAL (stable, no shuffle)
#
# [ADDED]
#   - CSV manifest: clip_id filtering + per-clip binary labels
#   - Per-segment labels (y) + one-hot labels (y_onehot) returned in batch
#   - max_segments cap (dataset-level)
#   - batch_size clamp so it never exceeds number of segments
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math
import random
import re
from collections import defaultdict

import csv
from typing import Iterable

import torch
import torch.nn.functional as F  # [ADDED] for one-hot labels
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import torch.distributed as dist

import lightning as L

_AUDIO_RE = re.compile(r"^(?P<clip>.+)_(?P<idx>\d{4})\.pt$")


# ============================================================
# [ADDED] Optional CSV manifest support
# - Filter to a provided set of clip_ids
# - Load binary labels per clip_id
# ============================================================
def _infer_clip_id_from_row(row: Dict[str, str]) -> Optional[str]:
    """Best-effort clip_id evaluation_for_detection_model from a CSV row.

    Priority:
      1) clip_id / clip_ids columns
      2) file / filename columns (strip dirs + extension)
    """
    for k in ("clip_id", "clip_ids", "clip", "id"):
        if k in row and row[k].strip():
            return row[k].strip()
    for k in ("file", "filename", "path"):
        if k in row and row[k].strip():
            p = Path(row[k].strip())
            # e.g., test/000001.mp4 -> 000001
            return p.stem
    return None


def _infer_label_from_row(row: Dict[str, str]) -> Optional[int]:
    for k in ("label", "y", "target", "class"):
        if k in row and row[k].strip() != "":
            v = int(float(row[k].strip()))
            if v not in (0, 1):
                raise ValueError(f"Binary label expected (0/1), got {v} from column '{k}'")
            return v
    return None


def _load_manifest_csv(csv_path: Union[str, Path], strict: bool = True) -> Tuple[List[str], Dict[str, int]]:
    """Return (clip_ids_in_order, labels_by_clip_id)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV manifest not found: {csv_path}")

    clip_ids: List[str] = []
    labels: Dict[str, int] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV manifest has no header: {csv_path}")

        for row in reader:
            cid = _infer_clip_id_from_row(row)
            if cid is None:
                if strict:
                    raise ValueError(f"Could not infer clip_id from row: {row}")
                continue

            y = _infer_label_from_row(row)
            if y is None and strict:
                raise ValueError(f"Could not infer binary label from row (clip_id={cid}): {row}")

            clip_ids.append(cid)
            if y is not None:
                labels[cid] = int(y)

    # Deduplicate but preserve order
    seen = set()
    clip_ids_unique: List[str] = []
    for cid in clip_ids:
        if cid not in seen:
            seen.add(cid)
            clip_ids_unique.append(cid)

    return clip_ids_unique, labels


# ============================================================
# [EXISTING] DDP-safe bucketed batch sampler (train)
# ============================================================
class DistributedBucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        *,
        lengths: List[int],
        batch_size: int,
        bucket_size: int = 8,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(None)
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.bucket_size = int(bucket_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

        self.epoch = 0
        self._build_buckets()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_buckets(self) -> None:
        pairs = list(enumerate(self.lengths))
        pairs.sort(key=lambda x: x[1])

        self.buckets = defaultdict(list)
        for idx, L_ in pairs:
            bucket_id = int(L_ // max(1, self.bucket_size))
            self.buckets[bucket_id].append(idx)

        self.bucket_ids = sorted(self.buckets.keys())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        all_batches = []
        for b in self.bucket_ids:
            idxs = list(self.buckets[b])
            if self.shuffle:
                rng.shuffle(idxs)

            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)

        for bi, batch in enumerate(all_batches):
            if (bi % self.num_replicas) == self.rank:
                yield batch

    def __len__(self) -> int:
        total_batches = 0
        for b in self.bucket_ids:
            n = len(self.buckets[b])
            total_batches += (n // self.batch_size) if self.drop_last else math.ceil(n / self.batch_size)
        return math.ceil(total_batches / max(1, self.num_replicas))


# ============================================================
# [EXISTING] Dataset: tensor-only .pt payloads
#   audio: float32 (n_mels, T_audio)
#   video: uint8   (3, T_video, H, W)
# ============================================================
class SegmentDataset(Dataset):
    def __init__(
        self,
        *,
        offline_root: Union[str, Path],
        batch_name: str,
        audio_dirname: str = "audio",
        video_dirname: str = "video_face_crops",
        map_location: str = "cpu",
        strict: bool = True,
        # ------------------------------------------------------------
        # [ADDED] CSV manifest support
        csv_path: Optional[Union[str, Path]] = None,
        max_segments: Optional[int] = None,
        strict_csv: bool = True,
    ) -> None:
        super().__init__()
        self.offline_root = Path(offline_root)
        self.batch_name = str(batch_name)

        self.batch_dir = self.offline_root / self.batch_name

        self.audio_root = self.batch_dir / audio_dirname
        self.video_root = self.batch_dir / video_dirname

        self.map_location = str(map_location)
        self.strict = bool(strict)

        # ============================================================
        # [ADDED] Optional CSV-based filtering + labels (per clip_id)
        # ============================================================
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.max_segments = int(max_segments) if max_segments is not None else None
        self.strict_csv = bool(strict_csv)

        self._clip_ids_filter: Optional[List[str]] = None
        self._clip_id_set: Optional[set] = None
        self._labels_by_clip_id: Dict[str, int] = {}

        if self.csv_path is not None:
            clip_ids, labels = _load_manifest_csv(self.csv_path, strict=self.strict_csv)
            self._clip_ids_filter = clip_ids
            self._clip_id_set = set(clip_ids)
            self._labels_by_clip_id = labels

        # (clip_id, seg_idx, audio_pt, video_pt, T_video)
        self.index: List[Tuple[str, int, Path, Path, int]] = []
        self._build_index()

        if not self.index:
            raise RuntimeError(f"No aligned (audio_pt, video_pt) pairs found under: {self.batch_dir}")

    def _build_index(self) -> None:
        if not self.audio_root.exists():
            raise FileNotFoundError(f"Missing audio root: {self.audio_root}")
        if not self.video_root.exists():
            raise FileNotFoundError(f"Missing video root: {self.video_root}")

        # [MODIFIED] Optional: iterate only clip_ids provided in CSV manifest
        if self._clip_ids_filter is not None:
            clip_dirs = [self.audio_root / cid for cid in self._clip_ids_filter]
        else:
            clip_dirs = [p for p in self.audio_root.iterdir() if p.is_dir()]

        for clip_dir in clip_dirs:
            if not clip_dir.exists() or not clip_dir.is_dir():
                if self.strict and self._clip_ids_filter is not None:
                    raise FileNotFoundError(f"Missing clip directory for clip_id='{clip_dir.name}': {clip_dir}")
                continue
            clip_id = clip_dir.name

            for a_pt in clip_dir.glob("*.pt"):
                m = _AUDIO_RE.match(a_pt.name)
                if not m or m.group("clip") != clip_id:
                    continue

                seg_idx = int(m.group("idx"))

                seg_dir = self.video_root / clip_id / f"seg_{seg_idx:04d}"
                v_pt = seg_dir / f"seg_{seg_idx:04d}.pt"

                if not v_pt.exists():
                    if self.strict:
                        continue
                    else:
                        continue

                v = torch.load(v_pt, map_location="cpu")
                if not isinstance(v, torch.Tensor) or v.ndim != 4:
                    if self.strict:
                        raise ValueError(f"Video pt is not a 4D Tensor: {v_pt} (got {type(v)})")
                    continue
                if v.shape[0] != 3:
                    if self.strict:
                        raise ValueError(f"Expected video shape (3,T,H,W) at {v_pt}, got {tuple(v.shape)}")
                    continue

                T_video = int(v.shape[1])
                self.index.append((clip_id, seg_idx, a_pt, v_pt, T_video))

        self.index.sort(key=lambda x: (x[0], x[1]))

        # [ADDED] Cap total number of segments (useful for quick runs / debugging)
        if self.max_segments is not None:
            self.index = self.index[: self.max_segments]

    def get_lengths(self) -> List[int]:
        return [int(x[4]) for x in self.index]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, object]:
        clip_id, seg_idx, a_pt, v_pt, T_video = self.index[i]

        # [ADDED] Optional label lookup (per clip_id) from CSV manifest
        y: Optional[int] = None
        if self._labels_by_clip_id:
            if clip_id not in self._labels_by_clip_id and self.strict_csv:
                raise KeyError(f"No label found in CSV for clip_id='{clip_id}'")
            y = int(self._labels_by_clip_id.get(clip_id, 0))

        mel = torch.load(a_pt, map_location=self.map_location)
        if not isinstance(mel, torch.Tensor) or mel.ndim != 2:
            raise ValueError(f"Expected audio mel (n_mels,T... Tensor at {a_pt}, got {type(mel)} {getattr(mel,'shape',None)}")
        if mel.dtype != torch.float32:
            mel = mel.float()

        v = torch.load(v_pt, map_location="cpu")
        if not isinstance(v, torch.Tensor) or v.ndim != 4:
            raise ValueError(f"Expected video Tensor (3,T,H,W) at {v_pt}, got {type(v)} {getattr(v,'shape',None)}")
        if v.dtype != torch.uint8:
            v = v.to(torch.uint8)

        return {
            "clip_id": clip_id,
            "seg_idx": int(seg_idx),
            "T_video": int(T_video),
            "audio": mel,            # (n_mels, T_audio)
            "video_u8_cthw": v,      # (3, T_video, H, W) uint8
            "y": y,  # [ADDED] None if CSV not provided
        }


def _pad_video_u8(video_u8_cthw: torch.Tensor, T_target: int) -> torch.Tensor:
    C, T, H, W = video_u8_cthw.shape
    if T == T_target:
        return video_u8_cthw
    if T > T_target:
        return video_u8_cthw[:, :T_target]
    pad = video_u8_cthw[:, -1:].repeat(1, T_target - T, 1, 1)
    return torch.cat([video_u8_cthw, pad], dim=1)


def _pad_audio_mel(mel: torch.Tensor, T_target: int) -> torch.Tensor:
    n_mels, T = mel.shape
    if T == T_target:
        return mel
    if T > T_target:
        return mel[:, :T_target]
    pad = mel[:, -1:].repeat(1, T_target - T)
    return torch.cat([mel, pad], dim=1)


def collate_segments_bucket_pad(items: List[Dict[str, object]]) -> Dict[str, object]:
    if not items:
        raise ValueError("Empty batch")

    T_video_max = max(int(it["T_video"]) for it in items)
    T_audio_max = max(int(it["audio"].shape[1]) for it in items)

    videos: List[torch.Tensor] = []
    audios: List[torch.Tensor] = []
    clip_ids: List[str] = []
    seg_idxs: List[int] = []
    T_v_list: List[int] = []
    T_a_list: List[int] = []
    y_list: List[Optional[int]] = []  # [ADDED]

    for it in items:
        v = it["video_u8_cthw"]
        a = it["audio"]

        videos.append(_pad_video_u8(v, T_video_max))
        audios.append(_pad_audio_mel(a, T_audio_max))

        clip_ids.append(str(it["clip_id"]))
        seg_idxs.append(int(it["seg_idx"]))
        T_v_list.append(int(it["T_video"]))
        T_a_list.append(int(a.shape[1]))

        y_list.append(it.get("y", None))  # [ADDED]

    # [ADDED] Labels: build y (B,) and y_onehot (B,2) if available
    any_label = any(y is not None for y in y_list)
    if any_label:
        y_tensor = torch.tensor([int(y) if y is not None else 0 for y in y_list], dtype=torch.long)
        y_onehot = F.one_hot(y_tensor, num_classes=2).to(torch.float32)
    else:
        y_tensor = None
        y_onehot = None

    return {
    "audio": torch.stack(audios, dim=0),           # (B, n_mels, T_audio_max)
    "video": torch.stack(videos, dim=0),           # (B, 3, T_video_max, H, W)
    "y": y_tensor,                                 # (B,) or None
    "y_onehot": y_onehot                          # (B,2) or None
    }



# ============================================================
# [PATCHED] Lightning DataModule: adds val split + val loader
# ============================================================
class SegmentDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        offline_root: Union[str, Path],
        batch_name: str,
        batch_size: int = 32,      # per GPU
        bucket_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = True,
        map_location: str = "cpu",
        seed: int = 123,
        val_split: float = 0.05,   # [ADDED] 5% validation by default
        val_batch_size: Optional[int] = None,  # [ADDED] defaults to train batch size
        # ------------------------------------------------------------
        # [ADDED] CSV manifest for clip_id filtering + labels
        csv_path: Optional[Union[str, Path]] = None,
        max_segments: Optional[int] = None,
        strict_csv: bool = True,
    ) -> None:
        super().__init__()
        self.offline_root = Path(offline_root)
        self.batch_name = str(batch_name)

        self.batch_size = int(batch_size)
        self.bucket_size = int(bucket_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.drop_last = bool(drop_last)
        self.map_location = str(map_location)
        self.seed = int(seed)

        self.val_split = float(val_split)  # [ADDED]
        self.val_batch_size = int(val_batch_size) if val_batch_size is not None else int(batch_size)  # [ADDED]

        # [ADDED] CSV manifest params passed into SegmentDataset
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.max_segments = int(max_segments) if max_segments is not None else None
        self.strict_csv = bool(strict_csv)

        # [ADDED] Effective (clamped) batch sizes computed in setup()
        self._effective_train_bs: Optional[int] = None
        self._effective_val_bs: Optional[int] = None

        self.ds_full: Optional[SegmentDataset] = None
        self.ds_train: Optional[Subset] = None
        self.ds_val: Optional[Subset] = None
        self._train_sampler: Optional[DistributedBucketBatchSampler] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_full = SegmentDataset(
            offline_root=self.offline_root,
            batch_name=self.batch_name,
            map_location=self.map_location,
            strict=True,
            # [ADDED] CSV filtering + labels
            csv_path=self.csv_path,
            max_segments=self.max_segments,
            strict_csv=self.strict_csv,
        )

        n = len(self.ds_full)
        n_val = int(round(n * self.val_split))
        n_val = max(1, n_val) if self.val_split > 0 else 0
        n_train = n - n_val

        # [ADDED] Deterministic split (important for thesis reproducibility)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).tolist()

        train_idx = perm[:n_train]
        val_idx = perm[n_train:] if n_val > 0 else []

        self.ds_train = Subset(self.ds_full, train_idx)
        self.ds_val = Subset(self.ds_full, val_idx) if n_val > 0 else None

        # ============================================================
        # [ADDED] Clamp batch sizes so they never exceed number of segments
        # This prevents 'drop_last' from producing 0 batches when n < batch_size.
        # ============================================================
        n_train_eff = len(self.ds_train)
        self._effective_train_bs = max(1, min(self.batch_size, n_train_eff))
        n_val_eff = len(self.ds_val) if self.ds_val is not None else 0
        self._effective_val_bs = max(1, min(self.val_batch_size, n_val_eff)) if n_val_eff > 0 else None

        drop_last_train = bool(self.drop_last and (n_train_eff >= self._effective_train_bs))

        # [MODIFIED] Train sampler lengths are from ds_full lengths but indexed by train subset
        full_lengths = self.ds_full.get_lengths()
        train_lengths = [full_lengths[i] for i in train_idx]

        self._train_sampler = DistributedBucketBatchSampler(
            lengths=train_lengths,
            batch_size=self._effective_train_bs,  # [MODIFIED] clamped
            bucket_size=self.bucket_size,
            drop_last=drop_last_train,  # [MODIFIED] safe when n < batch_size
            shuffle=True,
            seed=self.seed,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_full is not None and self.ds_train is not None
        assert self._train_sampler is not None

        # sampler yields indices relative to TRAIN subset
        train_subset = self.ds_train

        return DataLoader(
            train_subset,
            batch_sampler=self._train_sampler,
            num_workers=self.num_workers,
            collate_fn=collate_segments_bucket_pad,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.ds_val is None:
            return None
        return DataLoader(
            self.ds_val,
            batch_size=self._effective_val_bs if self._effective_val_bs is not None else self.val_batch_size,  # [MODIFIED]
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_segments_bucket_pad,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False,
        )

    def on_train_epoch_start(self) -> None:
        if self._train_sampler is not None and hasattr(self._train_sampler, "set_epoch"):
            self._train_sampler.set_epoch(int(self.trainer.current_epoch))
