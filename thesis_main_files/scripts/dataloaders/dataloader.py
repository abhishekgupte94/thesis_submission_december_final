# dataloader.py
"""
dataloader.py
=============

LightningDataModules for:
    - AVSegmentTokensDataset   (Swin / downstream)
    - AVTokenizerSegmentsDataset (AVTokenizer / AVSegmentTokenisationWrapper)

1) AVSegmentTokenDataModule
   ------------------------
   Produces batches of **tokenised** segments for Swin:

   Batch structure:
   {
       "audio_tokens": FloatTensor [B, Sa, D_a]
       "video_tokens": FloatTensor [B, Sv, D_v]
       "sample_id":    List[str]         len=B
       "video_id":     List[str]         len=B
       "segment_idx":  LongTensor [B]
       "label":        LongTensor [B]    (optional)
   }

2) AVTokenizerDataModule
   ----------------------
   Produces batches of **preprocessed segments** for AVTokenizer training:

   Collated batch structure (after custom collate_fn):
   {
       "mel_stack":          FloatTensor [N_tot, n_mels, T_a]
       "segment_tensors":    List[FloatTensor (S_j, 3, H, W)]  length N_tot_video_segments
       "segments_sec_audio": Optional[List[(float, float)]] length N_tot or None
       "segments_sec_video": Optional[List[(float, float)]] length N_tot or None
       "meta": {
           "batch_size": int,
           "items": [
               { "id": ..., "video_id": ..., "label": ..., ... },
               ...
           ]
       }
   }
"""

from typing import Any, Dict, List, Optional  # [MODIFIED] removed duplicate Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from scripts.dataloaders.dataset_storehouse import (
    AVSegmentTokensDataset,
    AVTokenizerSegmentsDataset,
)


# ---------------------------------------------------------------------------
# Small helper for loader kwargs (A100 vs Mac)
# ---------------------------------------------------------------------------

def _loader_kwargs(num_workers: int) -> Dict[str, Any]:
    """
    [ADDED] Choose sensible defaults for pin_memory / persistent_workers.

    - On A100 / GPU:
        * num_workers > 0 → pin_memory=True, persistent_workers=True
    - On Mac/CPU or when num_workers == 0:
        * pin_memory=False, persistent_workers=False

    The DataModule still allows you to override num_workers in __init__.
    """
    using_cuda = torch.cuda.is_available()
    if not using_cuda or num_workers == 0:
        return dict(
            pin_memory=False,
            persistent_workers=False,
        )
    else:
        return dict(
            pin_memory=True,
            persistent_workers=True,
        )


# ---------------------------------------------------------------------------
# 1) AVSegmentTokenDataModule (for Swin)
# ---------------------------------------------------------------------------

class AVSegmentTokenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        index_json_path: str,
        batch_size: int = 32,
        # num_workers: int = 8,  # Multi-GPU training
        num_workers: int = 0,    # MAC/ARM training (default)  # [MODIFIED] documented
        root_dir: Optional[str] = None,
        drop_last: bool = True,   # [ADDED] useful for contrastive-type training
    ):
        super().__init__()
        self.index_json_path = index_json_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.drop_last = drop_last  # [ADDED]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # Here we assume all offline tokenisation is already done (.pt files + index JSON).
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = AVSegmentTokensDataset(
                index_json_path=self.index_json_path,
                split="train",
                root_dir=self.root_dir,
            )
            self.val_dataset = AVSegmentTokensDataset(
                index_json_path=self.index_json_path,
                split="val",
                root_dir=self.root_dir,
            )

        if stage in (None, "test"):
            self.test_dataset = AVSegmentTokensDataset(
                index_json_path=self.index_json_path,
                split="test",
                root_dir=self.root_dir,
            )

    def _make_loader(self, ds, shuffle: bool, drop_last: bool):
        kw = _loader_kwargs(self.num_workers)  # [ADDED]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,       # [ADDED]
            **kw,                      # [ADDED] pin_memory / persistent_workers
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self):
        # For val/test, we typically keep the last (smaller) batch
        return self._make_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False, drop_last=False)


# ---------------------------------------------------------------------------
# 2) AVTokenizerDataModule (for AVTokenizer / AVSegmentTokenisationWrapper)
# ---------------------------------------------------------------------------

class AVTokenizerDataModule(pl.LightningDataModule):
    """
    DataModule for training the AVTokenizer / AVSegmentTokenisationWrapper.

    Uses AVTokenizerSegmentsDataset and a custom collate_fn that merges a
    batch of videos into one big pool of segments (suitable for
    encode_from_tensors).
    """
    def __init__(
        self,
        index_json_path: str,
        batch_size: int = 4,
        # num_workers: int = 8,  # Multi-GPU training
        num_workers: int = 0,    # MAC/ARM training (default)  # [MODIFIED]
        root_dir: Optional[str] = None,
        drop_last: bool = False,  # [ADDED] usually you want all clips here, but configurable
    ):
        super().__init__()
        self.index_json_path = index_json_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.drop_last = drop_last  # [ADDED]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # As above, assume .pt segments + index already exist.
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = AVTokenizerSegmentsDataset(
                index_json_path=self.index_json_path,
                split="train",
                root_dir=self.root_dir,
            )
            self.val_dataset = AVTokenizerSegmentsDataset(
                index_json_path=self.index_json_path,
                split="val",
                root_dir=self.root_dir,
            )

        if stage in (None, "test"):
            self.test_dataset = AVTokenizerSegmentsDataset(
                index_json_path=self.index_json_path,
                split="test",
                root_dir=self.root_dir,
            )

    # ----------------------- collate_fn for tokenizer -----------------------

    @staticmethod
    def _merge_segment_lists(
        lists: List[Optional[List[Any]]]
    ) -> Optional[List[Any]]:
        """
        Merge a list of segment lists into one list, or return None if any of
        the inputs is None.
        """
        if any(l is None for l in lists):
            return None
        merged: List[Any] = []
        for l in lists:
            merged.extend(l)
        return merged

    def _collate_tokenizer_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        batch: list of dataset items, each:
            {
                "mel_stack": (N_a_i, n_mels, T_a),
                "segment_tensors": List[Tensor(S_k, 3, H, W)],
                "segments_sec_audio": Optional[List[(start, end)]],
                "segments_sec_video": Optional[List[(start, end)]],
                "meta": Dict[str, Any],
            }

        Returns a single merged dict suitable for encode_from_tensors.
        """
        # 1) Concatenate mel_stacks along segment dimension
        mel_stacks = [b["mel_stack"] for b in batch]
        mel_stack = torch.cat(mel_stacks, dim=0)  # (N_tot, n_mels, T_a)

        # 2) Merge segment_tensors lists
        segment_lists = [b["segment_tensors"] for b in batch]
        segment_tensors = self._merge_segment_lists(segment_lists)

        # 3) Merge segments_sec_* lists (if present)
        seg_a_lists = [b["segments_sec_audio"] for b in batch]
        segments_sec_audio = self._merge_segment_lists(seg_a_lists)

        seg_v_lists = [b["segments_sec_video"] for b in batch]
        segments_sec_video = self._merge_segment_lists(seg_v_lists)

        # 4) Pack meta for each item
        meta_items = [b["meta"] for b in batch]
        meta: Dict[str, Any] = {
            "batch_size": len(batch),
            "items": meta_items,
        }

        return {
            "mel_stack": mel_stack,
            "segment_tensors": segment_tensors,
            "segments_sec_audio": segments_sec_audio,
            "segments_sec_video": segments_sec_video,
            "meta": meta,
        }

    # -------------------------- loaders ------------------------------------

    def _make_loader(self, ds, shuffle: bool, drop_last: bool):
        kw = _loader_kwargs(self.num_workers)  # [ADDED]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,          # [ADDED]
            collate_fn=self._collate_tokenizer_batch,
            **kw,                         # [ADDED]
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False, drop_last=False)
