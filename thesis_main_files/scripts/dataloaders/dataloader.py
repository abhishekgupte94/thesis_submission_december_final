# dataloader.py
# ============================================================
# [PATCHED] Segment DataModule with:
#   - Train/Val split (val_split)
#   - train_dataloader + val_dataloader
#   - Bucketing sampler for TRAIN only
#   - Simple sequential batches for VAL (stable, no shuffle)
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math
import random
import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import torch.distributed as dist

import lightning as L

_AUDIO_RE = re.compile(r"^(?P<clip>.+)_(?P<idx>\d{4})\.pt$")


# ============================================================
# [EXISTING] DDP-safe bucketed batch sampler (train)
# ============================================================
class DistributedBucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        *,
        bucket_size: int = 8,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.bucket_size = int(bucket_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

        self.buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, T in enumerate(self.lengths):
            b = (max(1, T) - 1) // self.bucket_size
            self.buckets[int(b)].append(idx)

        self.bucket_ids = sorted(self.buckets.keys())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        bucket_ids = self.bucket_ids[:]
        if self.shuffle:
            rng.shuffle(bucket_ids)

        all_batches: List[List[int]] = []
        for b in bucket_ids:
            idxs = self.buckets[b][:]
            if self.shuffle:
                rng.shuffle(idxs)

            end = (len(idxs) // self.batch_size) * self.batch_size if self.drop_last else len(idxs)
            for i in range(0, end, self.batch_size):
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
    ) -> None:
        super().__init__()
        self.offline_root = Path(offline_root)
        self.batch_name = str(batch_name)

        self.batch_dir = self.offline_root / self.batch_name
        self.audio_root = self.batch_dir / audio_dirname
        self.video_root = self.batch_dir / video_dirname

        self.map_location = str(map_location)
        self.strict = bool(strict)

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

        for clip_dir in self.audio_root.iterdir():
            if not clip_dir.is_dir():
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

    def get_lengths(self) -> List[int]:
        return [int(x[4]) for x in self.index]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, object]:
        clip_id, seg_idx, a_pt, v_pt, T_video = self.index[i]

        mel = torch.load(a_pt, map_location=self.map_location)
        if not isinstance(mel, torch.Tensor) or mel.ndim != 2:
            raise ValueError(f"Expected audio mel (n_mels,T) Tensor at {a_pt}, got {type(mel)} {getattr(mel,'shape',None)}")
        if mel.dtype != torch.float32:
            mel = mel.float()

        video_u8 = torch.load(v_pt, map_location="cpu")
        if not isinstance(video_u8, torch.Tensor) or video_u8.ndim != 4 or video_u8.shape[0] != 3:
            raise ValueError(
                f"Expected video (3,T,H,W) Tensor at {v_pt}, got {type(video_u8)} {getattr(video_u8, 'shape', None)}")
        if video_u8.dtype != torch.uint8:
            video_u8 = video_u8.to(torch.uint8)

        # ------------------------------------------------------------
        # [ADDED | SAFETY] Convert BGR -> RGB at load time
        # OpenCV-native crops are BGR; saved .pt is (3,T,H,W) uint8 with [B,G,R].
        # Convert once here to feed RGB to downstream model/transforms.
        # ------------------------------------------------------------
        video_u8 = video_u8[[2, 1, 0], ...].contiguous()

        return {
            "clip_id": clip_id,
            "seg_idx": seg_idx,
            "audio": mel,                 # (n_mels, T_audio) float32
            "video_u8_cthw": video_u8,     # (3, T_video, H, W) uint8
            "T_video": int(T_video),
        }


# ============================================================
# [EXISTING] Collate: pad within batch
# ============================================================
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

    videos = []
    audios = []
    clip_ids = []
    seg_idxs = []
    T_v_list = []
    T_a_list = []

    for it in items:
        v = it["video_u8_cthw"]
        a = it["audio"]

        videos.append(_pad_video_u8(v, T_video_max))
        audios.append(_pad_audio_mel(a, T_audio_max))

        clip_ids.append(str(it["clip_id"]))
        seg_idxs.append(int(it["seg_idx"]))
        T_v_list.append(int(it["T_video"]))
        T_a_list.append(int(a.shape[1]))

    return {
        "clip_ids": clip_ids,
        "seg_idxs": seg_idxs,
        "T_video": T_v_list,
        "T_audio": T_a_list,
        "video_u8_cthw": torch.stack(videos, dim=0),   # (B,3,Tv_max,H,W) uint8
        "audio": torch.stack(audios, dim=0),           # (B,n_mels,Ta_max) float32
        "Tv_max": int(T_video_max),                    # [KEPT] useful for logging
        "Ta_max": int(T_audio_max),                    # [KEPT] useful for logging
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

        self.ds_full: Optional[SegmentDataset] = None          # [ADDED]
        self.ds_train: Optional[Subset] = None                 # [MODIFIED]
        self.ds_val: Optional[Subset] = None                   # [ADDED]
        self._train_sampler: Optional[DistributedBucketBatchSampler] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_full = SegmentDataset(
            offline_root=self.offline_root,
            batch_name=self.batch_name,
            map_location=self.map_location,
            strict=True,
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

        # [MODIFIED] Train sampler lengths are from ds_full lengths but indexed by train subset
        full_lengths = self.ds_full.get_lengths()
        train_lengths = [full_lengths[i] for i in train_idx]

        self._train_sampler = DistributedBucketBatchSampler(
            lengths=train_lengths,
            batch_size=self.batch_size,
            bucket_size=self.bucket_size,
            drop_last=self.drop_last,
            shuffle=True,
            seed=self.seed,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_full is not None and self.ds_train is not None
        assert self._train_sampler is not None

        # [ADDED] Important: sampler yields indices relative to TRAIN subset, not full dataset
        # We therefore wrap the subset in a small view with __getitem__ using train_idx mapping.
        train_subset = self.ds_train  # Subset(ds_full, train_idx)

        return DataLoader(
            train_subset,
            batch_sampler=self._train_sampler,
            num_workers=self.num_workers,
            collate_fn=collate_segments_bucket_pad,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    # [ADDED] Validation DataLoader (no shuffle, stable)
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.ds_val is None:
            return None
        return DataLoader(
            self.ds_val,
            batch_size=self.val_batch_size,
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



