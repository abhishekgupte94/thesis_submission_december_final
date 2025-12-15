# dataset_storehouse.py
"""
dataset_storehouse.py
=====================

1) AVSegmentTokensDataset
   -----------------------
   Dataset for loading **segment-level tokenised audio/video** from offline .pt
   files (for Swin / downstream classification).

   Index JSON entries should look like:

   {
       "id": "video123_seg000",
       "tokens_path": "preprocessed_tokens/video123_seg000.pt",
       "video_id": "video123",
       "segment_idx": 0,
       "split": "train",
       "label": 1   # optional
   }

   Each tokens_path .pt file is expected to contain:

   {
       "audio_tokens": FloatTensor [Sa, D_a],
       "video_tokens": FloatTensor [Sv, D_v]
   }

2) AVTokenizerSegmentsDataset
   ---------------------------
   Dataset for loading **segment-level preprocessed features** to train the
   AVTokenizer (AVSegmentTokenisationWrapper).

   Index JSON entries should look like (AVSpeech offline exporter):

   {
       "id": "clip_0001",
       "audio_pt": "batch_001/audio/clip_0001_audio.pt",
       "video_pt": "batch_001/video/clip_0001_video.pt",
       "timestamps_csv": "AVSpeech_timestamps_csv/batch_001/clip_0001_words.csv",
       "split": "train",   # optional; if missing, all entries belong to every split
       "label": 1          # optional
   }

   The audio .pt file is expected to contain:

   {
       "mel_segments":   List[Tensor (n_mels, T_a)],
       "segments_sec":   List[(start, end)],
       # plus metadata like "audio_file", "num_segments", "num_words", "config", ...
   }

   The video .pt file is expected to contain:

   {
       "video_segments": List[Tensor (S_i, 3, H, W)],
       "segments_sec":   List[(start, end)],
       # plus metadata like "video_file", "num_segments", "num_words", "config", ...
   }

   One dataset item corresponds to **one video’s worth of segments**, which is
   the natural unit to feed into:

       AVSegmentTokenisationWrapper.encode_from_tensors(
           mel_stack=...,
           segment_tensors=...,
           segments_sec_audio=...,
           segments_sec_video=...,
       )
"""

import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helper to load "entries" from index JSON
# ---------------------------------------------------------------------------

def _load_index_entries(index_json_path: str) -> List[Dict[str, Any]]:
    """
    Load index entries from a JSON file.

    Supported formats
    -----------------
    1) List-of-dicts::

           [
               {"id": "video123_seg000", ...},
               {"id": "video123_seg001", ...},
               ...
           ]

    2) Dict with explicit "entries"::

           {"entries": [ ... ]}

    3) Dict mapping IDs to per-item dicts (used by AVSpeech offline exporter)::

           {
               "clip_id_1": {...},
               "clip_id_2": {...},
               ...
           }

       In this case we create a list of entries, injecting an "id" field
       from the dict key if it is not already present.
    """
    with open(index_json_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    entries: List[Dict[str, Any]]

    if isinstance(index_data, dict):
        if "entries" in index_data:
            entries = index_data["entries"]
        else:
            # Assume mapping id -> entry-dict
            entries = []
            for key, value in index_data.items():
                if not isinstance(value, dict):
                    continue
                entry = dict(value)
                entry.setdefault("id", key)
                entries.append(entry)
    else:
        # Assume it's already a list[dict]
        entries = index_data

    if not isinstance(entries, list):
        raise ValueError(
            f"Index file {index_json_path} must contain either a list of entries, "
            f"an 'entries' key, or a mapping of id -> entry-dict."
        )

    return entries


# ---------------------------------------------------------------------------
# 1) Existing: AVSegmentTokensDataset (for Swin / downstream)
# ---------------------------------------------------------------------------

class AVSegmentTokensDataset(Dataset):
    """
    One dataset item = one segment (already tokenised).

    Returns a dict:
        {
            "audio_tokens": FloatTensor [Sa, D_a],
            "video_tokens": FloatTensor [Sv, D_v],
            "sample_id":    str,
            "video_id":     str,
            "segment_idx":  int,
            "label":        LongTensor [] or not present
        }

    NOTE:
    -----
    - Device-agnostic: all tensors are loaded on CPU (map_location="cpu").
    - Lightning/DDP-safe: no internal randomness beyond index order.
    """

    def __init__(
        self,
        index_json_path: str,
        split: str = "train",
        root_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.index_json_path = index_json_path
        self.root_dir = root_dir

        all_entries = _load_index_entries(index_json_path)

        # If entries carry an explicit "split" field, respect it.
        # Otherwise, treat the index as single-split and use all entries
        # regardless of the requested split.
        if any("split" in e for e in all_entries):
            self.entries: List[Dict[str, Any]] = [
                e for e in all_entries if e.get("split", "train") == split
            ]
        else:
            self.entries = list(all_entries)

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

        audio_tokens = data["audio_tokens"]  # [Sa, D_a]
        video_tokens = data["video_tokens"]  # [Sv, D_v]

        # [ADDED] ensure float32, which is what Swin & downstream heads expect
        audio_tokens = audio_tokens.float()  # [ADDED]
        video_tokens = video_tokens.float()  # [ADDED]

        item: Dict[str, Any] = {
            "audio_tokens": audio_tokens,
            "video_tokens": video_tokens,
            "sample_id": entry["id"],
            "video_id": entry.get("video_id", entry["id"]),
            "segment_idx": entry.get("segment_idx", 0),
        }

        # Optional classification target (for fine-tune)
        label = entry.get("label", None)
        if label is not None:
            item["label"] = torch.tensor(label, dtype=torch.long)

        return item


# ---------------------------------------------------------------------------
# 2) AVTokenizerSegmentsDataset (for AVTokenizer / AVSegmentTokenisationWrapper)
# ---------------------------------------------------------------------------

class AVTokenizerSegmentsDataset(Dataset):
    """
    One dataset item = one video-equivalent segment bundle, ready for
    AVSegmentTokenisationWrapper.encode_from_tensors(...).

    Returns a dict:
        {
            "mel_stack":           FloatTensor [N_a, n_mels, T_a],
            "segment_tensors":     List[FloatTensor (S_i, 3, H, W)],
            "segments_sec_audio":  Optional[List[(float, float)]],
            "segments_sec_video":  Optional[List[(float, float)]],
            "meta":                Dict[str, Any]    # includes id, video_id, label, ...
        }

    NOTE:
    -----
    - Also fully device-agnostic (loads to CPU).
    - The DataModule handles batching & moves to GPU.
    """

    def __init__(
        self,
        index_json_path: str,
        split: str = "train",
        root_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.index_json_path = index_json_path
        self.root_dir = root_dir

        all_entries = _load_index_entries(index_json_path)

        # If entries carry an explicit "split" field, respect it.
        # Otherwise, treat the index as single-split and use all entries.
        if any("split" in e for e in all_entries):
            self.entries: List[Dict[str, Any]] = [
                e for e in all_entries if e.get("split", "train") == split
            ]
        else:
            self.entries = list(all_entries)

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
        """
        One dataset item = one video-equivalent bundle of segments.

        This implementation is designed to work with the AVSpeech
        offline exporter, which produces an index JSON where each
        entry contains *separate* audio and video .pt paths, e.g.:

            {
                "id": "clip_0001",
                "audio_pt": "batch_001/audio/clip_0001_audio.pt",
                "video_pt": "batch_001/video/clip_0001_video.pt",
                "timestamps_csv": "...",
                "num_segments_audio": ...,
                "num_segments_video": ...,
                "status": "ok",
                ...
            }

        The audio .pt contains:
            - "mel_segments":   List[Tensor (n_mels, T_a)]
            - "segments_sec":   List[(start, end)]
            - other metadata

        The video .pt contains:
            - "video_segments": List[Tensor (S_i, 3, H, W)]
            - "segments_sec":   List[(start, end)]
            - other metadata
        """
        entry = self.entries[idx]

        # Resolve and load audio payload
        audio_pt_path = self._resolve_path(entry["audio_pt"])
        audio_data = torch.load(audio_pt_path, map_location="cpu")

        mel_segments = audio_data["mel_segments"]       # List[Tensor (n_mels, T_a)]
        segments_sec_audio = audio_data.get("segments_sec", None)

        if isinstance(mel_segments, list) and len(mel_segments) > 0:
            mel_stack = torch.stack(mel_segments, dim=0).float()  # (N_a, n_mels, T_a)  # [MODIFIED]
        else:
            # Empty or unexpected => return an empty stack
            mel_stack = torch.empty(0, 0, 0, dtype=torch.float32)  # [MODIFIED]

        # Resolve and load video payload
        video_pt_path = self._resolve_path(entry["video_pt"])
        video_data = torch.load(video_pt_path, map_location="cpu")

        segment_tensors = video_data["video_segments"]   # List[Tensor (S_i, 3, H, W)]
        segments_sec_video = video_data.get("segments_sec", None)

        # Build meta dictionary. Start with index info, then enrich with
        # useful metadata from payloads for debugging/introspection.
        meta: Dict[str, Any] = {}

        meta["id"] = entry.get("id")
        meta["video_id"] = entry.get("video_id", entry.get("id"))

        # Counts from index (if present)
        for key in [
            "num_segments_audio",
            "num_segments_video",
            "num_words_audio",
            "num_words_video",
            "status",
            "proc_time_sec",
        ]:
            if key in entry:
                meta[key] = entry[key]

        # Keep a pointer to the original (relative) paths for debugging.
        meta["audio_pt"] = entry.get("audio_pt")
        meta["video_pt"] = entry.get("video_pt")
        meta["timestamps_csv"] = entry.get("timestamps_csv")

        # Optional classification target (if provided in index)
        label = entry.get("label", None)
        if label is not None:
            meta["label"] = label

        return {
            "mel_stack": mel_stack,
            "segment_tensors": segment_tensors,
            "segments_sec_audio": segments_sec_audio,
            "segments_sec_video": segments_sec_video,
            "meta": meta,
        }
