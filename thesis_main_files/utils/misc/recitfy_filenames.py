#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional


def _extract_clip_id_from_audio_name(name: str) -> Optional[str]:
    """
    Audio rule (as requested):
      - Find the substring 'train' (case-insensitive)
      - Take whatever comes immediately after the 'n' in 'train' (i.e., after 'train')
      - Until the extension dot (handled by Path(name).stem)

    Examples:
      trim_audio_train1.wav          -> "1"
      trim_audio_trainABC_01.wav     -> "ABC_01"
      TRAINhello-world.wav           -> "hello-world"
    """
    stem = Path(name).stem  # filename without extension
    m = re.search(r"(?i)train(.+)$", stem)  # everything after 'train'
    if not m:
        return None
    clip = m.group(1)
    return str(clip) if clip != "" else None


def _extract_clip_id_from_video_name(name: str) -> Optional[str]:
    """
    Video rule (kept the same):
      video_<clip>.mp4  -> "<clip>" (clip can be any non-empty string without dots)
    Examples:
      video_1.mp4       -> "1"
      video_ABC_01.mp4  -> "ABC_01"
    """
    stem = Path(name).stem
    m = re.match(r"(?i)video_(.+)$", stem)
    return str(m.group(1)) if m else None


def _build_index(directory: Path, kind: str) -> Dict[str, Path]:
    if kind not in {"audio", "video"}:
        raise ValueError("kind must be 'audio' or 'video'")

    idx: Dict[str, Path] = {}
    for p in directory.iterdir():
        if not p.is_file():
            continue

        clip_id = (
            _extract_clip_id_from_audio_name(p.name)
            if kind == "audio"
            else _extract_clip_id_from_video_name(p.name)
        )
        if clip_id is None:
            continue

        clip_id = str(clip_id)
        if clip_id in idx and idx[clip_id] != p:
            print(f"[WARN] Duplicate {kind} clip id '{clip_id}': '{idx[clip_id].name}' and '{p.name}' (keeping first)")
            continue

        idx[clip_id] = p

    return idx


def rename_av_files_from_json(
    audio_dir: str | Path,
    video_dir: str | Path,
    json_path: str | Path,
    *,
    audio_ext: str = ".wav",
    video_ext: str = ".mp4",
    dry_run: bool = True,
    allow_overwrite: bool = False,
) -> None:
    audio_dir = Path(audio_dir).expanduser().resolve()
    video_dir = Path(video_dir).expanduser().resolve()
    json_path = Path(json_path).expanduser().resolve()

    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video dir not found: {video_dir}")
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON must be an object/dict like {'<clip>': [[...]]}")

    audio_index = _build_index(audio_dir, "audio")
    video_index = _build_index(video_dir, "video")

    print(f"[INFO] Loaded {len(data)} JSON keys")
    print(f"[INFO] Indexed audio clips: {len(audio_index)} | video clips: {len(video_index)}")
    print(f"[INFO] dry_run={dry_run}, allow_overwrite={allow_overwrite}")

    renamed_audio = renamed_video = 0
    missing_audio = missing_video = 0

    for clip_key in data.keys():
        clip = str(clip_key)

        # Audio
        a_src = audio_index.get(clip)
        if a_src is None:
            missing_audio += 1
        else:
            a_dst = audio_dir / f"{clip}{audio_ext}"
            if a_src.name != a_dst.name:
                if a_dst.exists() and not allow_overwrite:
                    print(f"[SKIP] Audio target exists: {a_dst.name} (source: {a_src.name})")
                else:
                    print(f"[RENAME] Audio: {a_src.name} -> {a_dst.name}")
                    if not dry_run:
                        if a_dst.exists() and allow_overwrite:
                            a_dst.unlink()
                        a_src.rename(a_dst)
                    renamed_audio += 1

        # Video
        v_src = video_index.get(clip)
        if v_src is None:
            missing_video += 1
        else:
            v_dst = video_dir / f"{clip}{video_ext}"
            if v_src.name != v_dst.name:
                if v_dst.exists() and not allow_overwrite:
                    print(f"[SKIP] Video target exists: {v_dst.name} (source: {v_src.name})")
                else:
                    print(f"[RENAME] Video: {v_src.name} -> {v_dst.name}")
                    if not dry_run:
                        if v_dst.exists() and allow_overwrite:
                            v_dst.unlink()
                        v_src.rename(v_dst)
                    renamed_video += 1

    print("\n[SUMMARY]")
    print(f"  Renamed audio: {renamed_audio}")
    print(f"  Renamed video: {renamed_video}")
    print(f"  Missing audio matches: {missing_audio}")
    print(f"  Missing video matches: {missing_video}")


if __name__ == "__main__":
    rename_av_files_from_json(
        audio_dir="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/video_files/AVSpeech/audio",
        video_dir="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/video_files/AVSpeech/video",
        json_path="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/AVSpeech/AVSpeech_timestamp_json_for_offline_training/AVSpeech_timestamp_json_for_offline_training.json",
        dry_run=False,          # set False to actually rename
        allow_overwrite=True, # set True to overwrite existing <clip>.wav/mp4
    )
