#!/usr/bin/env python3
# utils/Imp_indexing_help/build_finetune_segment_paths_manifest.py

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

_AUDIO_RE = re.compile(r"^(?P<clip>.+)_(?P<idx>\d{4})\.pt$")


def _normalize_clip_id(x: str) -> str:
    x = str(x).strip().replace("\\", "/")
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
    return audio_96_path.with_name(audio_96_path.stem + "__2048" + audio_96_path.suffix)


# audio_root/<clip_id>/<clip_id>_0007.pt and <clip_id>_0007__2048.pt
def iter_base_audio96(audio_root: Path, *, debug: bool = False) -> Iterator[Tuple[str, int, Path]]:
    top_dirs = [p for p in audio_root.iterdir() if p.is_dir()]
    if debug:
        print(f"[DEBUG] audio_root={audio_root} clip_dirs={len(top_dirs)}")

    for clip_dir in top_dirs:
        clip_id = clip_dir.name
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
# [CHANGED] Build a clip_id -> video_clip_dir map by matching clip directories like audio.
# Supports:
#   video_root/<clip_id>/
#   video_root/<level1>/<clip_id>/
# --------------------------------------------------------------------------------------
def build_video_clip_map(video_root: Path, *, debug: bool = False) -> Dict[str, Path]:
    hits: Dict[str, list[Path]] = {}

    # direct level: video_root/<clip_id>/
    for p in video_root.iterdir():
        if p.is_dir():
            hits.setdefault(p.name, []).append(p)

    # one level deep: video_root/<level1>/<clip_id>/
    for level1 in video_root.iterdir():
        if not level1.is_dir():
            continue
        # skip if level1 is already a clip dir with seg_* inside? doesn't matter; we still scan children
        for p in level1.iterdir():
            if p.is_dir():
                hits.setdefault(p.name, []).append(p)

    # deterministic selection (and optional debug on collisions)
    out: Dict[str, Path] = {}
    collisions = 0
    for clip_id, paths in hits.items():
        paths_sorted = sorted({pp.resolve() for pp in paths})
        out[clip_id] = paths_sorted[0]
        if len(paths_sorted) > 1:
            collisions += 1
            if debug:
                print(f"[DEBUG][VIDEO] clip_id={clip_id} has {len(paths_sorted)} dirs; picking {paths_sorted[0]}")
                for extra in paths_sorted[:5]:
                    print(f"  - {extra}")

    if debug:
        print(f"[DEBUG] video_root={video_root} mapped_clips={len(out)} collisions={collisions}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-root", type=str, required=True)
    ap.add_argument("--batch-name", type=str, required=True)
    ap.add_argument("--audio-dirname", type=str, default="audio")
    ap.add_argument("--video-dirname", type=str, default="video_face_crops")
    ap.add_argument("--out-csv", type=str, default="segment_index_finetune.csv")
    ap.add_argument("--strict", action="store_true", default=False)

    ap.add_argument("--labels-csv", type=str, default=None)
    ap.add_argument("--labels-filename-col", type=str, default="filename")
    ap.add_argument("--labels-label-col", type=str, default="label")

    ap.add_argument("--debug", action="store_true", default=False)
    ap.add_argument("--debug-limit", type=int, default=25)
    args = ap.parse_args()

    batch_dir = Path(args.offline_root) / args.batch_name
    audio_root = batch_dir / args.audio_dirname
    video_root = batch_dir / args.video_dirname

    print(f"[INFO] batch_dir={batch_dir}")
    print(f"[INFO] audio_root={audio_root} exists={audio_root.exists()}")
    print(f"[INFO] video_root={video_root} exists={video_root.exists()}")

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

    # [CHANGED] build map once
    video_clip_map = build_video_clip_map(video_root, debug=args.debug)

    counters = Counter()
    verbose_left = args.debug_limit

    wrote = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "seg_idx", "audio_rel", "audio2048_rel", "video_rel", "label"])

        for clip_id, seg_idx, a96 in iter_base_audio96(audio_root, debug=args.debug):
            counters["audio_base_candidates"] += 1

            a2048 = _resolve_audio_2048_path(a96)
            if not a2048.exists():
                counters["skip_missing_audio2048"] += 1
                if args.strict:
                    raise FileNotFoundError(f"Missing audio_2048: {a2048}")
                continue

            v_clip_dir = video_clip_map.get(clip_id)
            if v_clip_dir is None:
                counters["skip_missing_video_clipdir"] += 1
                if args.debug and verbose_left > 0:
                    print(f"[DEBUG][VIDEO] no clip_dir for clip_id={clip_id} under {video_root}")
                    verbose_left -= 1
                if args.strict:
                    raise FileNotFoundError(f"Missing video clip dir for clip_id={clip_id} under {video_root}")
                continue

            # keep your original expected segment tensor location
            # [CHANGED] video is mp4 segments, not pt tensors
            # [CHANGED] video is mp4 segments, not pt tensors
            v_mp4 = v_clip_dir / f"seg_{seg_idx:04d}" / f"seg_{seg_idx:04d}.mp4"
            if not v_mp4.exists():
                if args.strict:
                    raise FileNotFoundError(f"Missing video mp4: {v_mp4}")
                continue

            v_rel = str(v_mp4.resolve().relative_to(batch_dir.resolve()))

            a96_rel = str(a96.resolve().relative_to(batch_dir.resolve()))
            a2048_rel = str(a2048.resolve().relative_to(batch_dir.resolve()))
            # v_rel = str(v_mp4.resolve().relative_to(batch_dir.resolve()))

            label = label_map.get(_normalize_clip_id(clip_id), "")
            if label == "":
                counters["label_missing_for_clip"] += 1

            w.writerow([clip_id, seg_idx, a2048_rel, a2048_rel, v_rel, label])
            wrote += 1

    print(f"[finetune_paths_manifest] wrote={wrote} -> {out_path}")
    print("[SUMMARY] counts:")
    for k in sorted(counters.keys()):
        print(f"  - {k}: {counters[k]}")


if __name__ == "__main__":
    main()
