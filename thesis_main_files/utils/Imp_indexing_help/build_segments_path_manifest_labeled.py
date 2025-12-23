#!/usr/bin/env python3
# build_segment_paths_manifest.py
#
# Build a fast manifest of (audio_pt, video_mp4) pairs WITHOUT ffprobe.
# Uses os.scandir() instead of glob for speed on NFS.
#
# Output CSV columns:
#   clip_id, seg_idx, audio_rel, video_rel, label

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Optional
from typing import Iterator, Optional, Tuple


_AUDIO_RE = re.compile(r"^(?P<clip>.+)_(?P<idx>\d{4})\.pt$")


# --------------------------------------------------------------------------------------
# [ADDED] Optional label lookup (clip_id -> label) from a metadata CSV
# --------------------------------------------------------------------------------------
def _normalize_clip_key(x: str) -> str:
    # clip_id can be "000001" or "000001.mp4" or "test/000001.mp4"
    x = x.strip()
    x = os.path.basename(x)
    if x.lower().endswith(".mp4"):
        x = x[:-4]
    return x


def load_label_map(
    csv_path: Optional[Path],
    *,
    filename_col: str = "filename",
    label_col: str = "label",
) -> Dict[str, str]:
    """Return dict: clip_key (stem) -> label (as string)."""
    if csv_path is None:
        return {}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"labels csv not found: {csv_path}")

    m: Dict[str, str] = {}
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if filename_col not in (r.fieldnames or []):
            raise KeyError(f"Missing column '{filename_col}' in labels csv: {csv_path}")
        if label_col not in (r.fieldnames or []):
            raise KeyError(f"Missing column '{label_col}' in labels csv: {csv_path}")
        for row in r:
            key = _normalize_clip_key(str(row[filename_col]))
            lbl = str(row[label_col]).strip()
            if key:
                m[key] = lbl
    return m


def iter_audio_pt_files(audio_root: Path) -> Iterator[Tuple[str, int, Path]]:
    """Yield (clip_id, seg_idx, audio_pt_path) for matching audio pt files."""
    # audio_root/<clip_id>/*.pt
    with os.scandir(audio_root) as it:
        for clip_ent in it:
            if not clip_ent.is_dir():
                continue
            clip_id = clip_ent.name
            clip_dir = Path(clip_ent.path)

            with os.scandir(clip_dir) as it2:
                for f_ent in it2:
                    if not f_ent.is_file():
                        continue
                    name = f_ent.name
                    if not name.endswith(".pt"):
                        continue
                    m = _AUDIO_RE.match(name)
                    if not m:
                        continue
                    # strict match: file prefix must equal directory clip_id
                    if m.group("clip") != clip_id:
                        continue
                    seg_idx = int(m.group("idx"))
                    yield clip_id, seg_idx, Path(f_ent.path)


def build_manifest(
    batch_dir: Path,
    *,
    audio_dirname: str,
    video_dirname: str,
    strict: bool,
) -> Iterator[Tuple[str, int, str, str]]:
    label_map = label_map or {}

    audio_root = batch_dir / audio_dirname
    video_root = batch_dir / video_dirname

    if not audio_root.exists():
        raise FileNotFoundError(f"Missing audio root: {audio_root}")
    if not video_root.exists():
        raise FileNotFoundError(f"Missing video root: {video_root}")

    for clip_id, seg_idx, a_pt in iter_audio_pt_files(audio_root):
        seg_dir = video_root / clip_id / f"seg_{seg_idx:04d}"
        v_mp4 = seg_dir / f"seg_{seg_idx:04d}.mp4"

        if not v_mp4.exists():
            if strict:
                continue
            else:
                continue

        # store relative paths under batch_dir (matches your cache philosophy)
        a_rel = str(a_pt.resolve().relative_to(batch_dir.resolve()))
        v_rel = str(v_mp4.resolve().relative_to(batch_dir.resolve()))
        label = label_map.get(_normalize_clip_key(clip_id), "")
        yield clip_id, seg_idx, a_rel, v_rel, label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-root", type=str, required=True)
    ap.add_argument("--batch-name", type=str, required=True)
    ap.add_argument("--audio-dirname", type=str, default="audio")
    ap.add_argument("--video-dirname", type=str, default="video_face_crops")
    ap.add_argument("--out-csv", type=str, default="segment_paths.csv")
    ap.add_argument("--strict", action="store_true", default=False)
    # [ADDED] label enrichment
    ap.add_argument("--labels-csv", type=str, default=None, help="CSV with columns 'filename' and 'label' to enrich manifest")
    ap.add_argument("--labels-filename-col", type=str, default="filename")
    ap.add_argument("--labels-label-col", type=str, default="label")
    args = ap.parse_args()

    batch_dir = Path(args.offline_root) / args.batch_name
    out_path = batch_dir / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # [ADDED] Load label map once (clip_id stem -> label)
    label_map = load_label_map(
        Path(args.labels_csv) if args.labels_csv else None,
        filename_col=args.labels_filename_col,
        label_col=args.labels_label_col,
    )

    rows = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "seg_idx", "audio_rel", "video_rel", "label"])
        for clip_id, seg_idx, a_rel, v_rel, label in build_manifest(
            batch_dir,
            audio_dirname=args.audio_dirname,
            video_dirname=args.video_dirname,
            strict=bool(args.strict),
            label_map=label_map,
        ):
            w.writerow([clip_id, seg_idx, a_rel, v_rel, label])
            rows += 1

    print(f"[paths_manifest] wrote={rows} -> {out_path}")


if __name__ == "__main__":
    main()