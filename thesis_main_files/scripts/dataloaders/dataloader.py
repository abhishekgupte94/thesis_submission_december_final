"""
dataloader.py
=============

LightningDataModule for AVSegmentTokensDataset.
Produces dict-style batches ready for the Swin wrapper.

Batch structure:

{
    "audio_tokens": FloatTensor [B, Sa, D_a]
    "video_tokens": FloatTensor [B, Sv, D_v]
    "sample_id":    list[str] len B
    "video_id":     list[str] len B
    "segment_idx":  LongTensor or list[int] len B
    "label":        LongTensor [B] (optional)
}
"""

from typing import Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset_storehouse import AVSegmentTokensDataset


class AVSegmentTokenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        index_json_path: str,
        batch_size: int = 32,
       # num_workers: int = 8, #Multi-GPU training
            num_workers: int = 0, #MAC/ARM training
        root_dir: Optional[str] = None,
    ):
        super().__init__()
        self.index_json_path = index_json_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir

    # Called once before training/validation/testing starts
    def setup(self, stage: Optional[str] = None):
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

        self.test_dataset = AVSegmentTokensDataset(
            index_json_path=self.index_json_path,
            split="test",
            root_dir=self.root_dir,
        )

    def _make_loader(self, ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            # pin_memory=True, # Multi-GPU training
            pin_memory = False, # MAC/ARM Training
            # persistent_workers=True, # Multi-GPU training
            persistent_workers=False #MAC/ARM training
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)
