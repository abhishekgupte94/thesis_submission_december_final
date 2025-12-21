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
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import torch.distributed as dist

import lightning as L

# ============================================================
# [ADDED] MP4 helpers (drop-in replacement for loading video .pt tensors)
#   - We keep the same downstream output format: uint8 (3, T, H, W) in RGB.
#   - We probe T_video during indexing so the existing bucketing sampler works.
# ============================================================

def _probe_num_frames_mp4_ffprobe(mp4_path: Path) -> Optional[int]:
    """Return number of frames for mp4_path using ffprobe (fast, no full decode).

    Returns None if ffprobe is unavailable or probing fails.
    """
    try:
        import subprocess

        # nb_read_frames is not guaranteed for all containers/codecs, but works for typical MP4/H264 exports.
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(mp4_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip().splitlines()
        # ffprobe may output 1 or 2 lines depending on fields availability.
        for line in reversed(out):
            line = line.strip()
            if line.isdigit():
                n = int(line)
                if n > 0:
                    return n
    except Exception:
        return None
    return None


def _read_mp4_to_u8_cthw(mp4_path: Path) -> torch.Tensor:
    """Decode MP4 to uint8 RGB tensor (3, T, H, W).

    Prefers torchvision (FFmpeg backend) if available; falls back to OpenCV.
    """
    # Try torchvision first (usually available in PyTorch video pipelines)
    try:
        from torchvision.io import read_video  # type: ignore

        # video: (T, H, W, C) in RGB, uint8 (most common); audio ignored
        video_thwc, _, _ = read_video(str(mp4_path), pts_unit="sec")
        if not isinstance(video_thwc, torch.Tensor) or video_thwc.ndim != 4 or video_thwc.shape[-1] != 3:
            raise RuntimeError(f"torchvision.read_video returned unexpected video tensor for {mp4_path}")
        video_u8_cthw = video_thwc.to(torch.uint8).permute(3, 0, 1, 2).contiguous()  # (3,T,H,W)
        return video_u8_cthw
    except Exception:
        pass

    # Fallback: OpenCV (BGR) -> convert to RGB -> tensor
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {mp4_path}")

        frames: List[np.ndarray] = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(frame_bgr)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames decoded from: {mp4_path}")

        # (T,H,W,C) uint8, convert BGR->RGB
        video_thwc = np.stack(frames, axis=0)[:, :, :, ::-1]
        video_u8_cthw = torch.from_numpy(video_thwc).to(torch.uint8).permute(3, 0, 1, 2).contiguous()
        return video_u8_cthw
    except Exception as e:
        raise RuntimeError(f"Failed to decode mp4: {mp4_path} (torchvision+opencv both failed): {e}")


_AUDIO_RE = re.compile(r"^(?P<clip>.+)_(?P<idx>\d{4})\.pt$")


# ============================================================
# [EXISTING] DDP-friendly bucketed sampler
# ============================================================
class BucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        bucket_size: int = 50,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 123,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.lengths = lengths
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
            b = int(T // self.bucket_size)
            self.buckets[b].append(idx)

        self.bucket_keys = sorted(self.buckets.keys())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)

        all_batches: List[List[int]] = []

        bucket_keys = list(self.bucket_keys)
        if self.shuffle:
            g.shuffle(bucket_keys)

        for b in bucket_keys:
            idxs = list(self.buckets[b])
            if self.shuffle:
                g.shuffle(idxs)

            for k in range(0, len(idxs), self.batch_size):
                batch = idxs[k : k + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            g.shuffle(all_batches)

        # DDP sharding at batch level
        all_batches = all_batches[self.rank :: self.num_replicas]
        return iter(all_batches)

    def __len__(self):
        total = 0
        for b in self.bucket_keys:
            n = len(self.buckets[b])
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += math.ceil(n / self.batch_size)

        return math.ceil(total / self.num_replicas)


# ============================================================
# [EXISTING] Dataset
# ============================================================
class SegmentDataset(Dataset):
    def __init__(
        self,
        offline_root: Union[str, Path],
        batch_name: str,
        *,
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

        # (clip_id, seg_idx, audio_pt, video_mp4, T_video)  # [MODIFIED] video path now points to *.mp4
        self.index: List[Tuple[str, int, Path, Path, int]] = []  # [UNCHANGED] Path now is mp4
        self._build_index()

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
                v_mp4 = seg_dir / f"seg_{seg_idx:04d}.mp4"  # [MODIFIED] load from mp4, not pt

                if not v_mp4.exists():
                    if self.strict:
                        continue
                    else:
                        continue

                # ------------------------------------------------------------
                # [MODIFIED] Probe number of frames without decoding (for bucketing)
                # ------------------------------------------------------------
                T_video_probe = _probe_num_frames_mp4_ffprobe(v_mp4)
                if T_video_probe is None:
                    if self.strict:
                        raise RuntimeError(f"Could not probe frame count with ffprobe for: {v_mp4}")
                    # Fallback: decode once (slower), but keeps behavior safe
                    v_u8 = _read_mp4_to_u8_cthw(v_mp4)
                    T_video_probe = int(v_u8.shape[1])

                T_video = int(T_video_probe)

                self.index.append((clip_id, seg_idx, a_pt, v_mp4, T_video))

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

        # ------------------------------------------------------------
        # [MODIFIED] Load video from MP4 (drop-in replacement for loading *.pt tensors)
        # Expect output: torch.Tensor uint8 (3, T, H, W) in RGB
        # ------------------------------------------------------------
        video_u8 = _read_mp4_to_u8_cthw(v_pt)

        if not isinstance(video_u8, torch.Tensor) or video_u8.ndim != 4 or video_u8.shape[0] != 3:
            raise ValueError(
                f"Expected decoded video uint8 Tensor (3,T,H,W) from {v_pt}, got {type(video_u8)} {getattr(video_u8,'shape',None)}"
            )
        if video_u8.dtype != torch.uint8:
            video_u8 = video_u8.to(torch.uint8)

        # NOTE:
        #   - Previously, video tensors were saved from OpenCV in BGR and we swapped to RGB.
        #   - Here, torchvision.read_video returns RGB already; OpenCV fallback explicitly converts to RGB.
        #   - So no channel swap is needed (keeps downstream RGB expectation).

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
def _pad_video_u8(video_u8_cthw: torch.Tensor, T: int) -> torch.Tensor:
    # video_u8_cthw: (3,T,H,W) uint8
    if video_u8_cthw.shape[1] == T:
        return video_u8_cthw
    pad_T = T - int(video_u8_cthw.shape[1])
    if pad_T <= 0:
        return video_u8_cthw[:, :T].contiguous()
    # pad on time dimension
    pad = torch.zeros((3, pad_T, video_u8_cthw.shape[2], video_u8_cthw.shape[3]), dtype=torch.uint8)
    return torch.cat([video_u8_cthw, pad], dim=1)


def _pad_audio(mel: torch.Tensor, T: int) -> torch.Tensor:
    # mel: (n_mels, T) float32
    if mel.shape[1] == T:
        return mel
    pad_T = T - int(mel.shape[1])
    if pad_T <= 0:
        return mel[:, :T].contiguous()
    pad = torch.zeros((mel.shape[0], pad_T), dtype=mel.dtype)
    return torch.cat([mel, pad], dim=1)


def collate_pad(batch: List[Dict[str, object]]) -> Dict[str, object]:
    clip_ids = [b["clip_id"] for b in batch]
    seg_idxs = torch.tensor([int(b["seg_idx"]) for b in batch], dtype=torch.long)

    audios = [b["audio"] for b in batch]  # type: ignore
    videos = [b["video_u8_cthw"] for b in batch]  # type: ignore

    max_Ta = max(int(a.shape[1]) for a in audios)
    max_Tv = max(int(v.shape[1]) for v in videos)

    audios_pad = torch.stack([_pad_audio(a, max_Ta) for a in audios], dim=0)   # (B, n_mels, T_a)
    videos_pad = torch.stack([_pad_video_u8(v, max_Tv) for v in videos], dim=0)  # (B,3,T_v,H,W)

    return {
        "clip_id": clip_ids,
        "seg_idx": seg_idxs,
        "audio": audios_pad,
        "video_u8_cthw": videos_pad,
        "T_audio": torch.tensor([int(a.shape[1]) for a in audios], dtype=torch.long),
        "T_video": torch.tensor([int(v.shape[1]) for v in videos], dtype=torch.long),
    }


# ============================================================
# [EXISTING] Lightning DataModule
# ============================================================
class SegmentDataModule(L.LightningDataModule):
    def __init__(
        self,
        offline_root: Union[str, Path],
        batch_name: str,
        *,
        audio_dirname: str = "audio_pt",
        video_dirname: str = "video_face_crops",
        batch_size: int = 4,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int = 123,
        bucket_size: int = 50,
        drop_last: bool = True,
        strict: bool = True,
    ) -> None:
        super().__init__()
        self.offline_root = Path(offline_root)
        self.batch_name = str(batch_name)

        self.audio_dirname = str(audio_dirname)
        self.video_dirname = str(video_dirname)

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.bucket_size = int(bucket_size)
        self.drop_last = bool(drop_last)
        self.strict = bool(strict)

        self.dataset: Optional[SegmentDataset] = None
        self.train_set: Optional[Subset] = None
        self.val_set: Optional[Subset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.dataset = SegmentDataset(
                self.offline_root,
                self.batch_name,
                audio_dirname=self.audio_dirname,
                video_dirname=self.video_dirname,
                strict=self.strict,
            )

            n = len(self.dataset)
            idxs = list(range(n))
            rng = random.Random(self.seed)
            rng.shuffle(idxs)

            n_val = int(round(n * self.val_split))
            val_idxs = idxs[:n_val]
            train_idxs = idxs[n_val:]

            self.train_set = Subset(self.dataset, train_idxs)
            self.val_set = Subset(self.dataset, val_idxs)

    def train_dataloader(self) -> DataLoader:
        assert self.dataset is not None and self.train_set is not None

        # get lengths for TRAIN subset indices
        all_lengths = self.dataset.get_lengths()
        train_lengths = [all_lengths[i] for i in self.train_set.indices]  # type: ignore

        sampler = BucketBatchSampler(
            train_lengths,
            batch_size=self.batch_size,
            bucket_size=self.bucket_size,
            drop_last=self.drop_last,
            shuffle=True,
            seed=self.seed,
        )

        # wrap sampler indices back to original dataset indices
        # each batch is indices into train_set; we map to original
        def _map_batch(batch: List[int]) -> List[int]:
            return [self.train_set.indices[j] for j in batch]  # type: ignore

        class _MappedBatchSampler(Sampler[List[int]]):
            def __iter__(self_inner):
                for batch in sampler:
                    yield _map_batch(batch)

            def __len__(self_inner):
                return len(sampler)

        return DataLoader(
            self.dataset,
            batch_sampler=_MappedBatchSampler(),
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_pad,
            persistent_workers= True
        )

    def val_dataloader(self) -> DataLoader:
        assert self.dataset is not None and self.val_set is not None

        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_pad,
            persistent_workers=True,
            drop_last=False,
        )
