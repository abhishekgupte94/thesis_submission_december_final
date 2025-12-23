#!/usr/bin/env python3
# utils/Imp_indexing_help/build_finetune_segment_paths_manifest.py
# ============================================================
# Build a per-batch manifest of segment paths for FINE-TUNE:
#   audio:  <audio_root>/<clip_id>/<clip_id>_0007.pt  and  <clip_id>_0007__2048.pt
#   video:  <video_root>/(<split>/)?<clip_id>/seg_0007/seg_0007.pt
#
# Output: <batch_dir>/<out_csv>
#   clip_id, seg_idx, audio96_rel, audio2048_rel, video_rel, label
# ============================================================

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


def resolve_video_clip_dir(video_root: Path, clip_id: str) -> Path:
    direct = video_root / clip_id
    if direct.exists():
        return direct

    candidates = [p for p in video_root.glob(f"*/{clip_id}") if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return sorted(candidates)[0]
    return direct


def iter_base_audio96(audio_root: Path, *, debug: bool = False) -> Iterator[Tuple[str, int, Path]]:
    """
    Yield (clip_id, seg_idx, audio96_pt_path) for base (non-__2048) audio files:
      audio_root/<clip_id>/<clip_id>_0007.pt
    """
    # Basic structure sanity
    top_entries = list(audio_root.iterdir()) if audio_root.exists() else []
    top_dirs = [p for p in top_entries if p.is_dir()]
    top_files = [p for p in top_entries if p.is_file()]
    if debug:
        print(f"[DEBUG] audio_root={audio_root}")
        print(f"[DEBUG] audio_root exists={audio_root.exists()} dirs={len(top_dirs)} files={len(top_files)}")
        if len(top_dirs) == 0:
            print("[DEBUG] WARNING: audio_root has NO sub-directories. Expected audio_root/<clip_id>/...")
            if len(top_files) > 0:
                print(f"[DEBUG] Example files in audio_root: {[p.name for p in top_files[:10]]}")

    for clip_dir in top_dirs:
        clip_id = clip_dir.name
        seen = 0
        kept = 0

        try:
            with os.scandir(clip_dir) as it2:
                for f_ent in it2:
                    if not f_ent.is_file():
                        continue
                    name = f_ent.name
                    if not name.endswith(".pt"):
                        continue

                    seen += 1

                    if name.endswith("__2048.pt"):
                        if debug:
                            print(f"[DEBUG][AUDIO] skip __2048: {clip_id}/{name}")
                        continue

                    m = _AUDIO_RE.match(name)
                    if not m:
                        if debug:
                            print(f"[DEBUG][AUDIO] regex_miss: {clip_id}/{name} (pattern expects *_0000.pt with 4 digits)")
                        continue

                    if m.group("clip") != clip_id:
                        if debug:
                            print(f"[DEBUG][AUDIO] clip_mismatch: dir={clip_id} file_clip={m.group('clip')} file={name}")
                        continue

                    seg_idx = int(m.group("idx"))
                    kept += 1
                    yield clip_id, seg_idx, Path(f_ent.path)

        except FileNotFoundError:
            if debug:
                print(f"[DEBUG][AUDIO] clip_dir vanished? {clip_dir}")
            continue

        if debug and seen > 0 and kept == 0:
            print(f"[DEBUG][AUDIO] clip_dir={clip_dir} had {seen} .pt files but kept 0 after filters.")
        if debug and seen == 0:
            print(f"[DEBUG][AUDIO] clip_dir={clip_dir} had 0 .pt files.")


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
    ap.add_argument("--debug-limit", type=int, default=50, help="limit verbose per-item debug prints")

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
    if args.debug:
        print(f"[DEBUG] label_map size={len(label_map)} (labels_csv={'none' if not args.labels_csv else args.labels_csv})")

    counters = Counter()
    verbose_left = args.debug_limit

    wrote = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "seg_idx", "audio96_rel", "audio2048_rel", "video_rel", "label"])

        found_any_audio = False

        for clip_id, seg_idx, a96 in iter_base_audio96(audio_root, debug=args.debug):
            found_any_audio = True
            counters["audio_base_candidates"] += 1

            a2048 = _resolve_audio_2048_path(a96)
            if not a2048.exists():
                counters["skip_missing_audio2048"] += 1
                if args.debug and verbose_left > 0:
                    print(f"[DEBUG][PAIR] missing a2048 for a96={a96} expected={a2048}")
                    verbose_left -= 1
                if args.strict:
                    raise FileNotFoundError(f"Missing audio_2048: {a2048}")
                continue

            v_clip_dir = resolve_video_clip_dir(video_root, clip_id)
            if not v_clip_dir.exists():
                counters["skip_missing_video_clipdir"] += 1
                if args.debug and verbose_left > 0:
                    print(f"[DEBUG][VIDEO] clip_dir not found for clip_id={clip_id} tried={video_root/clip_id} and {video_root}/*/{clip_id}")
                    verbose_left -= 1
                if args.strict:
                    raise FileNotFoundError(f"Missing video clip dir for clip_id={clip_id} under {video_root}")
                continue

            v_pt = v_clip_dir / f"seg_{seg_idx:04d}" / f"seg_{seg_idx:04d}.pt"
            if not v_pt.exists():
                counters["skip_missing_video_pt"] += 1
                if args.debug and verbose_left > 0:
                    print(f"[DEBUG][VIDEO] missing v_pt={v_pt} (clip_id={clip_id}, seg_idx={seg_idx})")
                    # print a hint: list a couple seg dirs
                    try:
                        seg_dirs = sorted([p.name for p in v_clip_dir.iterdir() if p.is_dir()])[:10]
                        print(f"[DEBUG][VIDEO] example seg dirs under {v_clip_dir}: {seg_dirs}")
                    except Exception:
                        pass
                    verbose_left -= 1
                if args.strict:
                    raise FileNotFoundError(f"Missing video pt: {v_pt}")
                continue

            a96_rel = str(a96.resolve().relative_to(batch_dir.resolve()))
            a2048_rel = str(a2048.resolve().relative_to(batch_dir.resolve()))
            v_rel = str(v_pt.resolve().relative_to(batch_dir.resolve()))

            label = label_map.get(_normalize_clip_id(clip_id), "")
            if label == "":
                counters["label_missing_for_clip"] += 1

            w.writerow([clip_id, seg_idx, a96_rel, a2048_rel, v_rel, label])
            wrote += 1
            counters["wrote"] += 1

    if not found_any_audio:
        print("[ERROR] Found ZERO audio base candidates. This usually means:")
        print("  - audio_root has no clip subdirs, OR")
        print("  - filenames don't match: <clip_id>_<4digit>.pt (e.g. 000001_0000.pt), OR")
        print("  - files are not .pt, OR")
        print("  - clip_dir name != file clip prefix")

    print(f"[finetune_paths_manifest] wrote={wrote} -> {out_path}")
    print("[SUMMARY] counts:")
    for k in sorted(counters.keys()):
        print(f"  - {k}: {counters[k]}")


if __name__ == "__main__":
    main()
