#!/usr/bin/env python3
# ============================================================
# match_and_flatten_dfdc_av.py
#
# What it does:
#   - Finds videos under: <DFDC>/video/dfdc_train_part_*/
#   - Finds audios under: <DFDC>/audio/dfdc_train_part_*/
#   - Matches by basename (stem): 000001.mp4 <-> 000001.wav (or other audio ext)
#   - Moves matched pairs into:
#       <DFDC>/Audio/   (audio destination)
#       <DFDC>/video/   (video destination; FLATTENED)
#   - If move succeeds, deletes now-empty part subdirs.
#
# Safety:
#   - Dry-run supported
#   - Never deletes non-empty folders (except .gitkeep cleanup)
# ============================================================

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ----------------------------
# Config / Extensions
# ----------------------------
VIDEO_EXTS: Set[str] = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTS: Set[str] = {".wav", ".m4a", ".aac", ".mp3", ".flac", ".ogg", ".opus"}


@dataclass
class MatchResult:
    stem: str
    video_path: Path
    audio_path: Path


def _iter_media_files(root: Path, exts: Set[str]) -> Iterable[Path]:
    """Recursively yield files under root with suffix in exts."""
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _build_stem_index(files: Iterable[Path]) -> Dict[str, List[Path]]:
    """
    Map: stem -> [paths...]
    (We keep a list in case duplicates exist in different subfolders.)
    """
    idx: Dict[str, List[Path]] = {}
    for f in files:
        idx.setdefault(f.stem, []).append(f)
    return idx


def _pick_one(paths: List[Path]) -> Path:
    """
    If multiple candidates exist for same stem, choose a deterministic one:
      - shortest path string length, then lexicographic.
    You can replace this with a duration-based pick if needed.
    """
    return sorted(paths, key=lambda p: (len(str(p)), str(p)))[0]


def match_audio_to_video(
    dfdc_root: Path,
    *,
    audio_parent_rel: str = "audio",
    video_parent_rel: str = "video",
) -> List[MatchResult]:
    """
    [STAGE 1] Match audio to video by stem.

    Assumes:
      - videos are under: <dfdc_root>/<video_parent_rel>/dfdc_train_part_*/...
      - audios are under: <dfdc_root>/<audio_parent_rel>/dfdc_train_part_*/...
    """
    audio_parent = dfdc_root / audio_parent_rel
    video_parent = dfdc_root / video_parent_rel

    video_files = list(_iter_media_files(video_parent, VIDEO_EXTS))
    audio_files = list(_iter_media_files(audio_parent, AUDIO_EXTS))

    v_idx = _build_stem_index(video_files)
    a_idx = _build_stem_index(audio_files)

    matches: List[MatchResult] = []
    for stem, v_paths in v_idx.items():
        if stem not in a_idx:
            continue
        v_path = _pick_one(v_paths)
        a_path = _pick_one(a_idx[stem])
        matches.append(MatchResult(stem=stem, video_path=v_path, audio_path=a_path))

    return matches


def _ensure_unique_dest(dest_dir: Path, filename: str) -> Path:
    """
    Avoid collisions when flattening:
      - if dest already exists, append __dupN before suffix
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if not dest.exists():
        return dest

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 1
    while True:
        cand = dest_dir / f"{stem}__dup{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def _safe_move(src: Path, dst: Path, *, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    shutil.move(str(src), str(dst))


def move_matched_pairs(
    matches: List[MatchResult],
    *,
    audio_out: Path,
    video_out: Path,
    dry_run: bool = True,
    log_every: int = 500,
) -> Tuple[int, int]:
    """
    [STAGE 2] Move matched audio/video into flat output folders.

    Returns: (moved_pairs, skipped_pairs)
    """
    moved = 0
    skipped = 0

    for i, m in enumerate(matches, start=1):
        # Determine destinations (preserve original filenames; de-dupe if needed)
        a_dst = _ensure_unique_dest(audio_out, m.audio_path.name)
        v_dst = _ensure_unique_dest(video_out, m.video_path.name)

        # If already at destination, skip cleanly
        if m.audio_path.resolve() == a_dst.resolve() and m.video_path.resolve() == v_dst.resolve():
            skipped += 1
            continue

        # Perform moves
        _safe_move(m.audio_path, a_dst, dry_run=dry_run)
        _safe_move(m.video_path, v_dst, dry_run=dry_run)

        # Verify (only if not dry_run)
        if not dry_run:
            if not a_dst.exists():
                raise RuntimeError(f"Audio move failed: {m.audio_path} -> {a_dst}")
            if not v_dst.exists():
                raise RuntimeError(f"Video move failed: {m.video_path} -> {v_dst}")

        moved += 1

        if log_every > 0 and (i % log_every == 0):
            print(f"[move] i={i}/{len(matches)} moved={moved} skipped={skipped}")

    return moved, skipped


def _dir_is_effectively_empty(d: Path) -> bool:
    """
    A directory is 'effectively empty' if it contains:
      - no files/dirs, OR
      - only .gitkeep files (any depth is NOT allowed; only direct children)
    """
    if not d.exists() or not d.is_dir():
        return False

    children = list(d.iterdir())
    if not children:
        return True

    # If anything besides .gitkeep exists, not empty
    for c in children:
        if c.is_dir():
            return False
        if c.name != ".gitkeep":
            return False
    return True


def cleanup_empty_part_dirs(
    dfdc_root: Path,
    *,
    audio_parent_rel: str = "audio",
    video_parent_rel: str = "video",
    dry_run: bool = True,
) -> int:
    """
    [STAGE 3] Delete now-empty part subdirs:
      <dfdc_root>/audio/dfdc_train_part_*/
      <dfdc_root>/video/dfdc_train_part_*/

    Also removes .gitkeep if it is the only remaining file in that dir.

    Returns: number of directories deleted
    """
    deleted = 0
    for parent_rel in (audio_parent_rel, video_parent_rel):
        parent = dfdc_root / parent_rel
        if not parent.exists():
            continue

        # Only target dfdc_train_part_* under each parent
        for part_dir in sorted(parent.glob("dfdc_train_part_*")):
            if not part_dir.is_dir():
                continue

            if _dir_is_effectively_empty(part_dir):
                # Remove .gitkeep first (if present)
                gitkeep = part_dir / ".gitkeep"
                if gitkeep.exists():
                    if dry_run:
                        print(f"[cleanup][dryrun] unlink {gitkeep}")
                    else:
                        gitkeep.unlink()

                if dry_run:
                    print(f"[cleanup][dryrun] rmdir  {part_dir}")
                else:
                    part_dir.rmdir()

                deleted += 1

    return deleted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfdc-root", type=str, required=True, help="Path to DFDC root (contains audio/ and video/)")
    ap.add_argument("--audio-out", type=str, default=None, help="Audio destination folder (default: <DFDC>/Audio)")
    ap.add_argument("--video-out", type=str, default=None, help="Video destination folder (default: <DFDC>/video)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without changing anything")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    dfdc_root = Path(args.dfdc_root).expanduser().resolve()
    if not dfdc_root.exists():
        raise FileNotFoundError(f"DFDC root not found: {dfdc_root}")

    # Defaults per your message:
    #   - move matched audio into: <DFDC>/Audio/
    #   - move matched video into: <DFDC>/video/   (flatten into the existing video dir)
    audio_out = Path(args.audio_out).expanduser().resolve() if args.audio_out else (dfdc_root / "Audio")
    video_out = Path(args.video_out).expanduser().resolve() if args.video_out else (dfdc_root / "video")

    print(f"[config] dfdc_root={dfdc_root}")
    print(f"[config] audio_out={audio_out}")
    print(f"[config] video_out={video_out}")
    print(f"[config] dry_run={args.dry_run}")

    # 1) Match
    matches = match_audio_to_video(dfdc_root)
    print(f"[match] videos↔audios matched={len(matches)}")

    if len(matches) == 0:
        print("[match] No matches found. Are stems identical between audio and video filenames?")
        return

    # 2) Move
    moved, skipped = move_matched_pairs(
        matches,
        audio_out=audio_out,
        video_out=video_out,
        dry_run=args.dry_run,
        log_every=args.log_every,
    )
    print(f"[move] moved_pairs={moved} skipped_pairs={skipped}")

    # 3) Cleanup part dirs (ONLY if moves are successful; dry-run prints)
    deleted_dirs = cleanup_empty_part_dirs(dfdc_root, dry_run=args.dry_run)
    print(f"[cleanup] deleted_part_dirs={deleted_dirs}")


if __name__ == "__main__":
    main()
