"""
dataset_storehouse.py
=====================

Dataset for loading segment-level tokenised audio/video from offline .pt files.

Each JSON entry should look like:

{
    "id": "video123_seg000",
    "tokens_path": "preprocessed_tokens/video123_seg000.pt",
    "video_id": "video123",
    "segment_idx": 0,
    "split": "train",
    "label": 1   # optional
}

The .pt file contains:
{
    "audio_tokens": Tensor [Sa, D_a],
    "video_tokens": Tensor [Sv, D_v]
}
"""

import os
import json
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset


class AVSegmentTokensDataset(Dataset):
    """
    One dataset item = one segment.

    Returns a dict:
        {
            "audio_tokens": FloatTensor [Sa, D_a],
            "video_tokens": FloatTensor [Sv, D_v],
            "sample_id": str,
            "video_id": str,
            "segment_idx": int,
            "label": LongTensor [] or None
        }
    """

    def __init__(
        self,
        index_json_path: str,
        split: str = "train",
        root_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.root_dir = root_dir

        # Load full index
        with open(index_json_path, "r") as f:
            all_entries: List[Dict[str, Any]] = json.load(f)

        # Filter by split
        self.entries = [e for e in all_entries if e["split"] == split]

        if len(self.entries) == 0:
            raise ValueError(
                f"No entries found for split='{split}' in index {index_json_path}"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_path(self, rel_path: str) -> str:
        if self.root_dir is not None:
            return os.path.join(self.root_dir, rel_path)
        return rel_path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]

        # ---- Load token file ----
        tokens_path = self._resolve_path(entry["tokens_path"])
        data = torch.load(tokens_path, map_location="cpu")

        audio_tokens = data["audio_tokens"]     # [Sa, D_a]
        video_tokens = data["video_tokens"]     # [Sv, D_v]

        item: Dict[str, Any] = {
            "audio_tokens": audio_tokens,
            "video_tokens": video_tokens,
            "sample_id": entry["id"],
            "video_id": entry["video_id"],
            "segment_idx": entry["segment_idx"],
        }

        # Optional classification target (for fine-tune)
        label = entry.get("label", None)
        if label is not None:
            item["label"] = torch.tensor(label, dtype=torch.long)

        return item
