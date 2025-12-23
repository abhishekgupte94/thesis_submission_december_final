#!/usr/bin/env python3
# utils/Imp_indexing_help/build_finetune_segment_paths_manifest.py
# ============================================================
# Build a fast per-batch manifest of segment paths for FINE-TUNE:
#   - audio_96 pt (base)      : <audio_root>/<clip_id>/<clip_id>_<seg>.pt
#   - audio_2048 pt (paired)  : <audio_root>/<clip_id>/<clip_id>_<seg>__2048.pt
#   - video pt tensor         : <video_root>/(<split>/)?<clip_id>/seg_0007/seg_0007.pt
#
# Output (written into <batch_dir>/<out_csv>):
#   clip_id, seg_idx, audio96_rel, audio2048_rel, video_rel, label
# ============================================================

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

_AUDIO_RE = re.compile(r"^(?P<clip>.+)_(?P<idx>\d{4})\.pt$")


# --------------------------------------------------------------------------------------
# Label map: labels_csv["filename"] is clip_id WITHOUT extension (clip-level labels)
# --------------------------------------------------------------------------------------
def _normalize_clip_id(x: str) -> str:
    x = str(x).strip().replace("\\", "/")
    x = os.path.basename(x)  # tolerate accidental paths
    if x.lower().endswith(".mp4"):
        x = x[:-4]
    return x


def load_label_map(
    csv_path: Optional[Path],
    *,
    filename_col: str = "filename",
    label_col: str = "label",
) -> Dict[str, str]:
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
            key = _normalize_clip_id(row[filename_col])
            lbl = str(row[label_col]).strip()
            if key:
                m[key] = lbl
    return m


def _resolve_audio_2048_path(audio_96_path: Path) -> Path:
    # <clip>_<seg>.pt  ->  <clip>_<seg>__2048.pt
    return audio_96_path.with_name(audio_96_path.stem + "__2048" + audio_96_path.suffix)


# --------------------------------------------------------------------------------------
# [CHANGED] Audio iterator: NESTED structure
#   audio_root/<clip_id>/<clip_id>_0007.pt
#   audio_root/<clip_id>/<clip_id>_0007__2048.pt
# --------------------------------------------------------------------------------------
def iter_base_audio96(audio_root: Path) -> Iterator[Tuple[str, int, Path]]:
    """Yield (clip_id, seg_idx, audio96_pt_path) for base (non-__2048) audio files inside audio_root/<clip_id>/."""
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
                    if name.endswith("__2048.pt"):
                        continue
                    m = _AUDIO_RE.match(name)
                    if not m:
                        continue
                    if m.group("clip") != clip_id:
                        continue
                    seg_idx = int(m.group("idx"))
                    yield clip_id, seg_idx, Path(f_ent.path)


# --------------------------------------------------------------------------------------
# Video clip dir resolver:
#   video_root/<clip_id>/...
#   OR video_root/<split>/<clip_id>/...
# (labels do NOT depend on split/segment)
# --------------------------------------------------------------------------------------
def resolve_video_clip_dir(video_root: Path, clip_id: str) -> Path:
    direct = video_root / clip_id
    if direct.exists():
        return direct

    candidates = [p for p in video_root.glob(f"*/{clip_id}") if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return sorted(candidates)[0]  # deterministic
    return direct  # will fail existence checks later


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-root", type=str, required=True)
    ap.add_argument("--batch-name", type=str, required=True)
    ap.add_argument("--audio-dirname", type=str, default="audio")
    ap.add_argument("--video-dirname", type=str, default="video_face_crops")
    ap.add_argument("--out-csv", type=str, default="segment_index_finetune.csv")
    ap.add_argument("--strict", action="store_true", default=False)

    ap.add_argument("--labels-csv", type=str, default=None, help="CSV with columns 'filename' and 'label'")
    ap.add_argument("--labels-filename-col", type=str, default="filename")
    ap.add_argument("--labels-label-col", type=str, default="label")
    args = ap.parse_args()

    batch_dir = Path(args.offline_root) / args.batch_name
    audio_root = batch_dir / args.audio_dirname
    video_root = batch_dir / args.video_dirname

    if not audio_root.exists():
        raise FileNotFoundError(f"Missing audio root: {audio_root}")
    if not video_root.exists():
        raise FileNotFoundError(f"Missing video root: {video_root}")

    out_path = batch_dir / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(
        Path(args.labels_csv) if args.labels_csv else None,
        filename_col=args.labels_filename_col,
        label_col=args.labels_label_col,
    )

    wrote = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "seg_idx", "audio96_rel", "audio2048_rel", "video_rel", "label"])

        for clip_id, seg_idx, a96 in iter_base_audio96(audio_root):
            a2048 = _resolve_audio_2048_path(a96)
            if not a2048.exists():
                if args.strict:
                    raise FileNotFoundError(f"Missing audio_2048: {a2048}")
                continue

            v_clip_dir = resolve_video_clip_dir(video_root, clip_id)
            v_pt = v_clip_dir / f"seg_{seg_idx:04d}" / f"seg_{seg_idx:04d}.pt"
            if not v_pt.exists():
                if args.strict:
                    raise FileNotFoundError(f"Missing video pt: {v_pt}")
                continue

            a96_rel = str(a96.resolve().relative_to(batch_dir.resolve()))
            a2048_rel = str(a2048.resolve().relative_to(batch_dir.resolve()))
            v_rel = str(v_pt.resolve().relative_to(batch_dir.resolve()))

            # clip-level label lookup
            label = label_map.get(_normalize_clip_id(clip_id), "")

            w.writerow([clip_id, seg_idx, a96_rel, a2048_rel, v_rel, label])
            wrote += 1

    print(f"[finetune_paths_manifest] wrote={wrote} -> {out_path}")


if __name__ == "__main__":
    main()
