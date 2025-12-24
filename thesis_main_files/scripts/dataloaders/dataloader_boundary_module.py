# dataloader_av_paths.py
# ============================================================
# [NEW | PATCHED] AUDIO+VIDEO PATH DATALOADER (Indexing + JSON cache)
#
# Purpose:
#   - Mirror your original loader skeleton (build_index -> JSON cache -> fallback)
#   - Index BOTH audio and video, returning only file paths (no preprocessing)
#   - Use CSV manifest if available, else cached JSON, else filesystem discovery
#
# Output per item:
#   {
#     "clip_id": str,
#     "seg_idx": int,
#     "video_path": Path,
#     "audio_path": Path,
#     "video_rel": str,   # relative to batch_dir
#     "audio_rel": str,   # relative to batch_dir
#     "y": Optional[int], # 0/1 if available
#   }
#
# CSV expectations (flexible):
#   A) clip_id,seg_idx,video_rel,audio_rel,label
#      - video_rel/audio_rel are relative to <batch_dir>
#   B) filename,label
#      - resolves:
#          video: <video_root>/<filename>.mp4
#          audio: <audio_root>/<filename>.wav
#
# Disk fallback (glob):
#   - video: <video_root>/**/*.mp4
#   - audio: <audio_root>/**/*.wav
#   - pairs by (clip_id, seg_idx) derived from filename pattern
#
# Notes:
#   - No padding, no mel, no tensors
#   - Your official repo preprocessor should run later in the evaluator/system
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
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Filename parsing:
#   000470.mp4
#   000470_0007.mp4
#   000470.wav
#   000470_0007.wav
# ============================================================
_AV_FILE_RE = re.compile(r"^(?P<clip>.+?)(?:_(?P<idx>\d+))?\.(?P<ext>mp4|wav)$", re.IGNORECASE)


def _load_labels_csv(csv_path: Path, *, strict: bool = True) -> Dict[Tuple[str, int], int]:
    """
    Supports:
      - clip_id, seg_idx, label
      - filename, label
    Returns {(clip_id, seg_idx): 0/1}
    """
    labels: Dict[Tuple[str, int], int] = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            if strict:
                raise ValueError(f"Labels CSV has no header: {csv_path}")
            return labels

        has_triplet = {"clip_id", "seg_idx", "label"}.issubset(set(reader.fieldnames))
        has_filename = {"filename", "label"}.issubset(set(reader.fieldnames))

        if not (has_triplet or has_filename):
            if strict:
                raise ValueError(
                    f"Labels CSV must have either (clip_id,seg_idx,label) OR (filename,label). "
                    f"Got headers: {reader.fieldnames}"
                )
            return labels

        for row in reader:
            if row is None:
                continue

            if has_triplet:
                clip_id = (row.get("clip_id") or "").strip()
                seg_idx_str = (row.get("seg_idx") or "").strip()
                label_str = (row.get("label") or "").strip()
                if not clip_id or not seg_idx_str or not label_str:
                    if strict:
                        raise ValueError(f"Bad row (missing fields): {row}")
                    continue
                seg_idx = int(seg_idx_str)
                y = int(label_str)
            else:
                filename = (row.get("filename") or "").strip()
                label_str = (row.get("label") or "").strip()
                if not filename or not label_str:
                    if strict:
                        raise ValueError(f"Bad row (missing fields): {row}")
                    continue
                stem = filename[:-4] if filename.lower().endswith((".mp4", ".pt")) else filename
                clip_id = stem
                seg_idx = 0
                y = int(label_str)

            if y not in (0, 1):
                if strict:
                    raise ValueError(f"Bad label value (must be 0/1): {row}")
                continue

            labels[(clip_id, seg_idx)] = y

    return labels


class AVPathsDataset(Dataset):
    """
    Index tuple:
      (clip_id, seg_idx, video_path, audio_path, y_or_None)
    """

    def __init__(
        self,
        *,
        offline_root: Optional[Path] = None,
        batch_name: Optional[str] = None,
        video_root: Optional[Path] = None,
        audio_root: Optional[Path] = None,
        strict: bool = False,
        max_items: Optional[int] = None,
        index_csv: Optional[Path] = None,
        labels_csv: Optional[Path] = None,
        strict_labels: bool = True,
    ) -> None:
        super().__init__()

        self.strict = bool(strict)
        self.max_items = max_items

        # [SKELETON] Resolve batch_dir
        self.offline_root = Path(offline_root) if offline_root is not None else None
        self.batch_name = str(batch_name) if batch_name is not None else None

        if video_root is not None or audio_root is not None:
            if video_root is None or audio_root is None:
                raise ValueError("If passing roots directly, pass BOTH video_root and audio_root.")
            self.video_root = Path(video_root).expanduser().resolve()
            self.audio_root = Path(audio_root).expanduser().resolve()
            # batch_dir becomes common parent for relpaths
            self.batch_dir = Path(os.path.commonpath([self.video_root, self.audio_root])).resolve()
        else:
            if self.offline_root is None or self.batch_name is None:
                raise ValueError("Provide either (video_root+audio_root) OR (offline_root+batch_name).")
            self.batch_dir = (self.offline_root / self.batch_name).expanduser().resolve()
            self.video_root = (self.batch_dir / "video_face_crops").resolve()
            self.audio_root = (self.batch_dir / "audio").resolve()

        if not self.video_root.exists():
            raise FileNotFoundError(f"Missing video root: {self.video_root}")
        if not self.audio_root.exists():
            raise FileNotFoundError(f"Missing audio root: {self.audio_root}")

        # optional CSVs
        self.index_csv = Path(index_csv).expanduser().resolve() if index_csv else None
        self.labels_csv = Path(labels_csv).expanduser().resolve() if labels_csv else None
        self.strict_labels = bool(strict_labels)

        self._labels_by_key: Dict[Tuple[str, int], int] = {}
        if self.labels_csv is not None and self.labels_csv.exists():
            self._labels_by_key = _load_labels_csv(self.labels_csv, strict=self.strict_labels)

        # index: (clip_id, seg_idx, video_path, audio_path, y)
        self.index: List[Tuple[str, int, Path, Path, Optional[int]]] = []
        self._build_index()

    # -------------------------
    # Cache paths
    # -------------------------
    def _index_cache_path(self) -> Path:
        return self.batch_dir / ".av_paths_index_cache_v1.json"

    def _index_cache_lock_path(self) -> Path:
        return self.batch_dir / ".av_paths_index_cache_v1.lock"

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

    def _try_load_index_cache(self, cache_path: Path) -> bool:
        try:
            raw = json.loads(cache_path.read_text())
            if not isinstance(raw, list):
                return False

            loaded: List[Tuple[str, int, Path, Path, Optional[int]]] = []
            for row in raw:
                if not isinstance(row, dict):
                    return False

                clip_id = str(row.get("clip_id"))
                seg_idx = int(row.get("seg_idx"))

                video_rel = row.get("video_rel")
                audio_rel = row.get("audio_rel")
                if not isinstance(video_rel, str) or not isinstance(audio_rel, str):
                    return False

                video_path = (self.batch_dir / video_rel).resolve()
                audio_path = (self.batch_dir / audio_rel).resolve()

                if not video_path.exists() or not audio_path.exists():
                    return False

                y = row.get("label", None)
                y_val: Optional[int]
                if y is None:
                    y_val = None
                else:
                    y_int = int(y)
                    if y_int not in (0, 1):
                        return False
                    y_val = y_int

                if self._labels_by_key:
                    y_val = self._labels_by_key.get((clip_id, seg_idx), y_val)

                loaded.append((clip_id, seg_idx, video_path, audio_path, y_val))

            loaded.sort(key=lambda x: (x[0], x[1]))
            if self.max_items is not None:
                loaded = loaded[: self.max_items]
            self.index = loaded
            return True
        except Exception:
            return False

    def _save_index_cache(self, cache_path: Path) -> None:
        try:
            payload = []
            for (clip_id, seg_idx, vpath, apath, y) in self.index:
                payload.append(
                    {
                        "clip_id": clip_id,
                        "seg_idx": int(seg_idx),
                        "video_rel": str(vpath.relative_to(self.batch_dir)),
                        "audio_rel": str(apath.relative_to(self.batch_dir)),
                        "label": None if y is None else int(y),
                    }
                )
            cache_path.write_text(json.dumps(payload))
        except Exception:
            pass

    # -------------------------
    # CSV manifest path index
    # -------------------------
    def _try_load_index_from_csv(self, csv_path: Path) -> bool:
        """
        Supports:
          A) clip_id,seg_idx,video_rel,audio_rel,label
          B) filename,label  (pairs roots)
        """
        try:
            with csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return False

                fields = set(reader.fieldnames)
                has_a = {"clip_id", "seg_idx", "video_rel", "audio_rel"}.issubset(fields)
                has_b = {"filename"}.issubset(fields)

                if not (has_a or has_b):
                    return False

                loaded: List[Tuple[str, int, Path, Path, Optional[int]]] = []

                for row in reader:
                    if row is None:
                        continue

                    if has_a:
                        clip_id = (row.get("clip_id") or "").strip()
                        seg_idx_str = (row.get("seg_idx") or "").strip()
                        video_rel = (row.get("video_rel") or "").strip()
                        audio_rel = (row.get("audio_rel") or "").strip()
                        label_str = (row.get("label") or "").strip() if "label" in fields else ""

                        if not clip_id or not seg_idx_str or not video_rel or not audio_rel:
                            return False
                        seg_idx = int(seg_idx_str)

                        vpath = (self.batch_dir / video_rel).resolve()
                        apath = (self.batch_dir / audio_rel).resolve()

                        y: Optional[int] = None
                        if label_str:
                            y_int = int(label_str)
                            if y_int not in (0, 1):
                                return False
                            y = y_int

                    else:
                        filename = (row.get("filename") or "").strip()
                        label_str = (row.get("label") or "").strip() if "label" in fields else ""
                        if not filename:
                            return False

                        stem = filename
                        if stem.lower().endswith(".mp4"):
                            stem = stem[:-4]
                        if stem.lower().endswith(".pt"):
                            stem = stem[:-4]

                        clip_id = stem
                        seg_idx = 0

                        vpath = (self.video_root / f"{stem}.mp4").resolve()
                        apath = (self.audio_root / f"{stem}.pt").resolve()

                        y = None
                        if label_str:
                            y_int = int(label_str)
                            if y_int not in (0, 1):
                                return False
                            y = y_int

                    if not vpath.exists() or not apath.exists():
                        if self.strict:
                            raise FileNotFoundError(f"Missing pair: video={vpath} audio={apath}")
                        continue

                    if self._labels_by_key:
                        y = self._labels_by_key.get((clip_id, seg_idx), y)

                    loaded.append((clip_id, int(seg_idx), vpath, apath, y))

                loaded.sort(key=lambda x: (x[0], x[1]))
                if self.max_items is not None:
                    loaded = loaded[: self.max_items]
                self.index = loaded
                return True
        except Exception:
            return False

    # -------------------------
    # Disk discovery fallback
    # -------------------------
    def _discover_from_disk(self) -> None:
        videos = list(self.video_root.rglob("*.mp4"))
        audios = list(self.audio_root.rglob("*.pt"))

        # Build audio map by key
        audio_map: Dict[Tuple[str, int], Path] = {}
        for ap in audios:
            m = _AV_FILE_RE.match(ap.name)
            if not m:
                continue
            clip = m.group("clip")
            idx = m.group("idx")
            seg_idx = int(idx) if idx is not None else 0
            audio_map[(clip, seg_idx)] = ap.resolve()

        # Pair videos with audio
        for vp in videos:
            m = _AV_FILE_RE.match(vp.name)
            if not m:
                continue
            clip = m.group("clip")
            idx = m.group("idx")
            seg_idx = int(idx) if idx is not None else 0

            ap = audio_map.get((clip, seg_idx))
            if ap is None:
                if self.strict:
                    raise FileNotFoundError(f"Missing audio for: {vp}")
                continue

            y = None
            if self._labels_by_key:
                y = self._labels_by_key.get((clip, seg_idx), None)

            self.index.append((clip, seg_idx, vp.resolve(), ap, y))

        self.index.sort(key=lambda x: (x[0], x[1]))
        if self.max_items is not None:
            self.index = self.index[: self.max_items]

    # -------------------------
    # Build index: cache -> csv -> disk
    # -------------------------
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

            if self.index_csv is not None and self.index_csv.exists():
                if self._try_load_index_from_csv(self.index_csv):
                    self._save_index_cache(cache_path)
                    return

            self._discover_from_disk()
            self._save_index_cache(cache_path)

        finally:
            if got_lock:
                self._release_lock(lock_path)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, object]:
        clip_id, seg_idx, vpath, apath, y = self.index[i]
        if y is None and self.strict_labels and self._labels_by_key:
            raise KeyError(f"Missing label for (clip_id={clip_id}, seg_idx={seg_idx}) in labels_csv")

        return {
            "clip_id": clip_id,
            "seg_idx": int(seg_idx),
            "video_path": vpath,
            "audio_path": apath,
            "video_rel": str(vpath.relative_to(self.batch_dir)),
            "audio_rel": str(apath.relative_to(self.batch_dir)),
            "y": y,
        }


def collate_av_paths(items: List[Dict[str, object]]) -> Dict[str, object]:
    clip_ids = [str(it["clip_id"]) for it in items]
    seg_idxs = [int(it["seg_idx"]) for it in items]
    video_paths = [str(it["video_path"]) for it in items]
    audio_paths = [str(it["audio_path"]) for it in items]
    video_rels = [str(it["video_rel"]) for it in items]
    audio_rels = [str(it["audio_rel"]) for it in items]

    y_list = [it.get("y", None) for it in items]
    y_tensor = None
    if not all(y is None for y in y_list):
        import torch
        y_tensor = torch.tensor([int(yy) if yy is not None else 0 for yy in y_list], dtype=torch.long)

    return {
        "clip_ids": clip_ids,
        "seg_idxs": seg_idxs,
        "video_paths": video_paths,
        "audio_paths": audio_paths,
        "video_rels": video_rels,
        "audio_rels": audio_rels,
        "y": y_tensor,
    }


@dataclass
class AVPathsDataModuleConfig:
    offline_root: Optional[Path] = None
    batch_name: Optional[str] = None
    video_root: Optional[Path] = None
    audio_root: Optional[Path] = None

    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True

    strict: bool = False
    max_items: Optional[int] = None
    index_csv: Optional[Path] = None
    labels_csv: Optional[Path] = None
    strict_labels: bool = True

    val_split: float = 0.05
    seed: int = 123
    drop_last: bool = False


class AVPathsDataModule(pl.LightningDataModule):
    def __init__(self, cfg: AVPathsDataModuleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._train = None
        self._val = None

    def setup(self, stage: Optional[str] = None) -> None:
        ds = AVPathsDataset(
            offline_root=self.cfg.offline_root,
            batch_name=self.cfg.batch_name,
            video_root=self.cfg.video_root,
            audio_root=self.cfg.audio_root,
            strict=self.cfg.strict,
            max_items=self.cfg.max_items,
            index_csv=self.cfg.index_csv,
            labels_csv=self.cfg.labels_csv,
            strict_labels=self.cfg.strict_labels,
        )

        import torch
        n = len(ds)
        n_val = max(1, int(round(n * float(self.cfg.val_split)))) if n > 0 else 0
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
            collate_fn=collate_av_paths,
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
            collate_fn=collate_av_paths,
            drop_last=False,
        )

