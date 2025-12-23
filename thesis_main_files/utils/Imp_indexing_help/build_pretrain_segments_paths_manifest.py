#!/usr/bin/env python3
# utils/Imp_indexing_help/build_finetune_segment_paths_manifest.py
# ============================================================
# Build a fast per-batch manifest of segment paths for FINE-TUNE:
#   - audio_96 pt (base)
#   - audio_2048 pt (paired by name)
#   - video pt tensor
#
# Output (written into <batch_dir>/segment_paths_finetune.csv):
#   clip_id, seg_idx, audio96_rel, audio2048_rel, video_rel
# ============================================================

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Iterator, Tuple

_AUDIO_RE = re.compile(r"^(?P<clip>.+)_(?P<idx>\d{4})\.pt$")


def _resolve_audio_2048_path(audio_96_path: Path) -> Path:
    return audio_96_path.with_name(audio_96_path.stem + "__2048" + audio_96_path.suffix)


def iter_base_audio96(audio_root: Path) -> Iterator[Tuple[str, int, Path]]:
    """Yield (clip_id, seg_idx, audio96_pt_path) for base audio_96 files only."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-root", type=str, required=True)
    ap.add_argument("--batch-name", type=str, required=True)
    ap.add_argument("--audio-dirname", type=str, default="audio")
    ap.add_argument("--video-dirname", type=str, default="video_face_crops")
    ap.add_argument("--out-csv", type=str, default="segment_paths_finetune.csv")  # [CHANGED]
    ap.add_argument("--strict", action="store_true", default=False)
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

    wrote = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "seg_idx", "audio96_rel", "audio2048_rel", "video_rel"])

        for clip_id, seg_idx, a96 in iter_base_audio96(audio_root):
            a2048 = _resolve_audio_2048_path(a96)
            if not a2048.exists():
                if args.strict:
                    raise FileNotFoundError(f"Missing audio_2048: {a2048}")
                continue

            v_pt = video_root / clip_id / f"seg_{seg_idx:04d}" / f"seg_{seg_idx:04d}.pt"
            if not v_pt.exists():
                if args.strict:
                    raise FileNotFoundError(f"Missing video pt: {v_pt}")
                continue

            a96_rel = str(a96.resolve().relative_to(batch_dir.resolve()))
            a2048_rel = str(a2048.resolve().relative_to(batch_dir.resolve()))
            v_rel = str(v_pt.resolve().relative_to(batch_dir.resolve()))

            w.writerow([clip_id, seg_idx, a96_rel, a2048_rel, v_rel])
            wrote += 1

    print(f"[finetune_paths_manifest] wrote={wrote} -> {out_path}")


if __name__ == "__main__":
    main()
