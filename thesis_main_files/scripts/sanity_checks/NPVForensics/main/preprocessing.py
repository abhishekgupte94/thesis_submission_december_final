# preprocessing.py  (sanity checks for AudioPreprocessorNPV & VideoPreprocessorNPV)

"""
Sanity checks for the *new*:
  1) AudioPreprocessorNPV
  2) VideoPreprocessorNPV

Features:
- MemoryGuard to keep things safe on a 16GB macOS machine.
- Monkey-patch of AudioPreprocessorNPV._load_audio_file to use soundfile
  instead of torchaudio (avoids torchcodec/FFmpeg issues on Mac).
- Direct sanity checks (shapes/stats) for audio & video preprocessors.
- JSON-backed offline-save sanity that uses the *final* save helpers:

    AudioPreprocessorNPV.process_and_save_from_timestamps_csv_segmentlocal
    VideoPreprocessorNPV.process_and_save_from_timestamps_csv_segmentlocal

  by converting your JSON {file_stem: [[start, end], ...]} into a temporary
  Whisper-like CSV with "start,end" columns.
"""

from pathlib import Path
from typing import Sequence, List, Union, Optional

import os
import time
import types
import json
import csv

import torch
import soundfile as sf
import psutil

# ---------- ADJUST THESE IMPORT PATHS TO MATCH YOUR PROJECT ----------
from scripts.preprocessing.audio.AudioPreprocessorNPV import (
    AudioPreprocessorNPV,
    AudioPreprocessorConfig,
)
from scripts.preprocessing.video.VideoPreprocessorNPV import (
    VideoPreprocessorNPV,
    VideoPreprocessorConfig,
)


# ====================================================================
# MEMORY GUARD (SANITY-ONLY)
# ====================================================================


class MemoryGuard:
    """
    Simple, reliable macOS-safe memory guard.

    - Checks process RSS memory (resident set size).
    - Optionally checks system-wide available memory.
    - If threshold exceeded → raises MemoryError or returns False.

    Recommended threshold for 16GB RAM:
        8–10 GB for safety.
    """

    def __init__(
        self,
        max_process_gb: float = 8.0,
        min_system_available_gb: float = 2.0,
        throws: bool = False,
    ):
        self.max_process = max_process_gb * (1024**3)
        self.min_system_free = min_system_available_gb * (1024**3)
        self.throws = throws

    def _get_process_rss(self) -> int:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss

    def _get_system_available(self) -> int:
        vm = psutil.virtual_memory()
        return vm.available

    def check(self) -> bool:
        """
        Returns True if safe, otherwise raises or returns False.
        """
        rss = self._get_process_rss()
        avail = self._get_system_available()

        if rss > self.max_process or avail < self.min_system_free:
            msg = (
                f"[MemoryGuard] Unsafe memory conditions:\n"
                f"  - RSS: {rss / (1024**3):.2f} GB "
                f"(limit {self.max_process / (1024**3):.2f} GB)\n"
                f"  - Available: {avail / (1024**3):.2f} GB "
                f"(min {self.min_system_free / (1024**3):.2f} GB)\n"
            )
            if self.throws:
                raise MemoryError(msg)
            else:
                print(msg)
                return False

        return True


MEM_GUARD = MemoryGuard(
    max_process_gb=8.0,
    min_system_available_gb=2.0,
    throws=False,
)


def memsafe(stage: str) -> bool:
    """
    Convenience helper for sanity tests.

    Returns False and prints a message if memory is unsafe,
    so callers can early-return gracefully.
    """
    ok = MEM_GUARD.check()
    if not ok:
        print(f"[SAFE EXIT] Memory conditions unsafe at stage: {stage}. Skipping the rest.")
        return False
    return True


# ====================================================================
# HELPER: PATCH AUDIO LOADER TO USE SOUNDFILE (MAC-SAFE)
# ====================================================================


def patch_audio_loader_to_soundfile(ap: AudioPreprocessorNPV) -> None:
    """
    Monkey-patch ap._load_audio_file to use soundfile instead of torchaudio.

    Used only in this sanity script on your Mac.
    """

    def _load_audio_file_sf(self, path: Union[str, Path]):
        """
        Replacement for AudioPreprocessorNPV._load_audio_file used ONLY
        in this sanity script.

        Uses soundfile to read the waveform and returns a tensor of shape
        (1, T) and the sample rate.
        """
        path_str = str(path)
        print(f"[AUDIO][PATCH] Loading via soundfile: {path_str}")

        try:
            wav_np, sr = sf.read(path_str, dtype="float32")  # (T,) or (T, C)
        except Exception as e:
            print(f"[AUDIO][PATCH][ERROR] soundfile.read failed on: {path_str}")
            print(f"[AUDIO][PATCH][ERROR] Exception type: {type(e).__name__}")
            print(f"[AUDIO][PATCH][ERROR] Exception: {e}")
            raise

        # Convert to mono and channel-first (1, T)
        if wav_np.ndim == 1:
            wav_mono = wav_np  # (T,)
        else:
            # (T, C) -> mono (T,)
            wav_mono = wav_np.mean(axis=1)

        wav = torch.from_numpy(wav_mono)[None, :]  # (1, T)
        return wav, sr

    ap._load_audio_file = types.MethodType(_load_audio_file_sf, ap)


# ====================================================================
# HELPER: JSON → TEMP CSV (FOR FINAL SAVE HELPERS)
# ====================================================================


def build_temp_csv_from_json(
    json_path: Union[str, Path],
    file_stem: str,
    temp_csv_path: Union[str, Path],
) -> Path:
    """
    Convert a JSON {file_stem: [[start, end], ...], ...} into a small
    Whisper-like CSV with columns: start,end for a single file.

    Example JSON:
        {
            "trim_audio_train100": [[0.2, 0.9], [1.0, 1.9]],
            "video_83": [[0.2, 0.9], [1.0, 1.9]]
        }

    Returns the Path to the created CSV.

    Raises:
        FileNotFoundError if json_path does not exist
        KeyError if file_stem not found in JSON
        ValueError if the JSON format is not as expected
    """
    json_path = Path(json_path)
    temp_csv_path = Path(temp_csv_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Timestamps JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if file_stem not in data:
        raise KeyError(f"Key '{file_stem}' not found in {json_path}")

    word_times = data[file_stem]  # expected: List[List[float, float]]

    # Basic validation
    if not isinstance(word_times, list) or any(
        (not isinstance(p, (list, tuple))) or len(p) != 2 for p in word_times
    ):
        raise ValueError(
            f"JSON entry for '{file_stem}' must be List[[start, end], ...], "
            f"got (first 5): {word_times[:5]}"
        )

    temp_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with temp_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end"])
        max_files = 0
        for s, e in word_times:
            if max_files == 5:
                break
            writer.writerow([float(s), float(e)])
            max_files = max_files + 1

    print(f"[TS_JSON→CSV] Built temp CSV for '{file_stem}' at: {temp_csv_path}")
    return temp_csv_path


# ====================================================================
# SANITY: AUDIO PREPROCESSOR (DIRECT)
# ====================================================================


def sanity_audio_preprocessor(
    audio_path: Union[str, Path],
    word_times: Sequence[Sequence[float]],
) -> None:
    """
    Sanity check for AudioPreprocessorNPV *only* (direct API).

    - Monkey-patches _load_audio_file to use soundfile.
    - Runs full-utterance mel + segment-local mel.
    """
    print("\n=== [SANITY] AudioPreprocessorNPV ===")

    audio_path = Path(audio_path)
    print(f"[AUDIO] Target path: {audio_path}")
    print(f"[AUDIO] exists? {audio_path.exists()}")
    print(f"[AUDIO] is_file? {audio_path.is_file()}")

    if not audio_path.exists() or not audio_path.is_file():
        print("[AUDIO][ERROR] The audio file does not exist at this path.")
        return

    if not os.access(str(audio_path), os.R_OK):
        print("[AUDIO][ERROR] The audio file exists but is not readable (permissions?).")
        return

    if not memsafe("before audio preprocessor init"):
        return

    cfg = AudioPreprocessorConfig()
    ap = AudioPreprocessorNPV(cfg)
    patch_audio_loader_to_soundfile(ap)

    print(f"[AUDIO] Using config: {cfg}")
    print(f"[AUDIO] File: {audio_path}")

    # ---- Full-utterance mel -------------------------------------------------
    if not memsafe("before full-utterance audio processing"):
        return

    t0 = time.time()
    mel_full: torch.Tensor = ap.process_audio_file(audio_path)
    t1 = time.time()

    print(f"[AUDIO] Full-utterance mel shape: {tuple(mel_full.shape)}")
    print(
        f"[AUDIO] dtype={mel_full.dtype}, "
        f"min={mel_full.min().item():.4f}, max={mel_full.max().item():.4f}"
    )
    print(f"[AUDIO] Full mel computed in {t1 - t0:.3f} s")

    # ---- Segment-local mel segments -----------------------------------------
    if not memsafe("before segment-local audio processing"):
        return

    t0 = time.time()
    mel_segments, segments_sec = ap.process_file_with_word_segments_segmentlocal(
        path=audio_path,
        word_times=word_times,
    )
    t1 = time.time()

    num_segments = len(mel_segments)
    print(f"[AUDIO] #segments: {num_segments}")
    print(f"[AUDIO] segments_sec (first 5): {segments_sec[:5]}")
    print(f"[AUDIO] Segment-local mel computed in {t1 - t0:.3f} s")

    for idx, mel_seg in enumerate(mel_segments[:3]):
        print(f"  [AUDIO] seg[{idx}] shape: {tuple(mel_seg.shape)}")

    if num_segments == 0:
        print("[AUDIO] WARNING: No segments produced. Check word_times / timestamps.")


# ====================================================================
# SANITY: VIDEO PREPROCESSOR (DIRECT)
# ====================================================================


def sanity_video_preprocessor(
    video_path: Union[str, Path],
    word_times: Sequence[Sequence[float]],
    target_clip_duration: float,
) -> None:
    """
    Sanity check for VideoPreprocessorNPV *only*.

    - Uses the tensor-returning segment-local API.
    """
    print("\n=== [SANITY] VideoPreprocessorNPV ===")

    video_path = Path(video_path)
    print(f"[VIDEO] Target path: {video_path}")
    print(f"[VIDEO] exists? {video_path.exists()}")
    print(f"[VIDEO] is_file? {video_path.is_file()}")

    if not video_path.exists() or not video_path.is_file():
        print("[VIDEO][ERROR] The video file does not exist at this path.")
        return

    if not os.access(str(video_path), os.R_OK):
        print("[VIDEO][ERROR] The video file exists but is not readable (permissions?).")
        return

    if not memsafe("before video preprocessor init"):
        return

    vcfg = VideoPreprocessorConfig()
    vp = VideoPreprocessorNPV(vcfg)

    print(f"[VIDEO] Using config: {vcfg}")
    print(f"[VIDEO] File: {video_path}")
    print(f"[VIDEO] target_clip_duration={target_clip_duration:.3f} s")

    if not memsafe("before segment-local video processing"):
        return

    t0 = time.time()
    segment_tensors, segments_sec = vp.process_video_file_with_word_segments_tensor(
        video_path=str(video_path),
        cropped_root="",  # not used in tensor API
        word_times=word_times,
        target_clip_duration=target_clip_duration,
        video_id=video_path.stem,
        keep_full_when_no_face=True,
    )
    t1 = time.time()

    num_segments = len(segment_tensors)
    print(f"[VIDEO] #segments: {num_segments}")
    print(f"[VIDEO] segments_sec (first 5): {segments_sec[:5]}")
    print(f"[VIDEO] Segment-local video computed in {t1 - t0:.3f} s")

    for idx, seg_tensor in enumerate(segment_tensors[:3]):
        if not isinstance(seg_tensor, torch.Tensor):
            print(f"  [VIDEO] seg[{idx}] is not a Tensor, type={type(seg_tensor)}")
            continue
        print(f"  [VIDEO] seg[{idx}] shape: {tuple(seg_tensor.shape)}")
        if seg_tensor.numel() > 0:
            print(
                f"    stats: min={seg_tensor.min().item():.4f}, "
                f"max={seg_tensor.max().item():.4f}"
            )

    if num_segments == 0:
        print("[VIDEO] WARNING: No segments produced. Check word_times / timestamps.")


# ====================================================================
# SANITY: AUDIO OFFLINE SAVE (FINAL HELPER, JSON-BACKED)
# ====================================================================


def sanity_audio_offline_save(
    audio_path: Union[str, Path],
    timestamps_json_path: Union[str, Path],
    out_pt_path: Union[str, Path],
    log_csv_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Sanity check for:

        AudioPreprocessorNPV.process_and_save_from_timestamps_csv_segmentlocal

    But our ground-truth timestamps are stored in a JSON mapping:

        { "file_stem": [[start, end], ...], ... }

    So we:
      1) Load the JSON
      2) Extract the list for the relevant file_stem (audio_path.stem)
      3) Write a temporary CSV with start,end
      4) Call the *final* audio save helper with that CSV
      5) Optionally inspect the saved .pt payload
    """
    print("\n=== [SANITY] Audio OFFLINE SAVE (JSON-backed) ===")

    audio_path = Path(audio_path)
    timestamps_json_path = Path(timestamps_json_path)
    out_pt_path = Path(out_pt_path)
    log_csv_path = Path(log_csv_path) if log_csv_path is not None else None

    file_stem = audio_path.stem
    print(f"[AUDIO/OFFLINE] audio_path      = {audio_path}")
    print(f"[AUDIO/OFFLINE] timestamps_json = {timestamps_json_path}")
    print(f"[AUDIO/OFFLINE] file_stem       = {file_stem}")
    print(f"[AUDIO/OFFLINE] out_pt_path     = {out_pt_path}")
    print(f"[AUDIO/OFFLINE] log_csv_path    = {log_csv_path}")

    if not timestamps_json_path.exists():
        print("[AUDIO/OFFLINE][ERROR] timestamps JSON does not exist — aborting.")
        return

    if not memsafe("before audio offline save init"):
        return

    # 1) Build a temporary CSV just for this file
    temp_csv_path = out_pt_path.with_suffix(".words_tmp.csv")
    try:
        temp_csv_path = build_temp_csv_from_json(
            json_path=timestamps_json_path,
            file_stem=file_stem,
            temp_csv_path=temp_csv_path,
        )
    except Exception as e:
        print(f"[AUDIO/OFFLINE][ERROR] Failed to build temp CSV from JSON: {e}")
        return

    # 2) Run the final offline save helper (using the temp CSV)
    cfg = AudioPreprocessorConfig()
    ap = AudioPreprocessorNPV(cfg)
    patch_audio_loader_to_soundfile(ap)

    num_segments, num_words = ap.process_and_save_from_timestamps_csv_segmentlocal(
        audio_path=audio_path,
        timestamps_csv_path=temp_csv_path,
        out_pt_path=out_pt_path,
        log_csv_path=log_csv_path,
    )

    print(f"[AUDIO/OFFLINE] Saved {num_segments} segments from {num_words} words to {out_pt_path}")

    # 3) Inspect the saved payload
    if not out_pt_path.exists():
        print("[AUDIO/OFFLINE][WARN] out_pt_path does not exist after save.")
        return

    payload = torch.load(out_pt_path)
    print(f"[AUDIO/OFFLINE] payload keys: {list(payload.keys())}")
    print(f"[AUDIO/OFFLINE] num_segments (payload): {payload.get('num_segments')}")
    print(f"[AUDIO/OFFLINE] num_words    (payload): {payload.get('num_words')}")

    mel_segments = payload.get("mel_segments")
    if mel_segments and isinstance(mel_segments, list):
        first = mel_segments[0]
        if isinstance(first, torch.Tensor):
            print(f"[AUDIO/OFFLINE] first mel segment shape: {tuple(first.shape)}")

    # Optional: clean up temp CSV
    # temp_csv_path.unlink(missing_ok=True)


# ====================================================================
# SANITY: VIDEO OFFLINE SAVE (FINAL HELPER, JSON-BACKED)
# ====================================================================


def sanity_video_offline_save(
    video_path: Union[str, Path],
    timestamps_json_path: Union[str, Path],
    out_pt_path: Union[str, Path],
    log_csv_path: Optional[Union[str, Path]] = None,
    target_clip_duration: Optional[float] = None,
) -> None:
    """
    Sanity check for:

        VideoPreprocessorNPV.process_and_save_from_timestamps_csv_segmentlocal

    Using a shared JSON timestamps file:

        { "file_stem": [[start, end], ...], ... }

    We:
      1) Build a Whisper-like temp CSV for this video
      2) Call the *final* video save helper with that CSV
      3) Inspect the saved .pt payload
    """
    print("\n=== [SANITY] Video OFFLINE SAVE (JSON-backed) ===")

    video_path = Path(video_path)
    timestamps_json_path = Path(timestamps_json_path)
    out_pt_path = Path(out_pt_path)
    log_csv_path = Path(log_csv_path) if log_csv_path is not None else None

    file_stem = video_path.stem
    print(f"[VIDEO/OFFLINE] video_path      = {video_path}")
    print(f"[VIDEO/OFFLINE] timestamps_json = {timestamps_json_path}")
    print(f"[VIDEO/OFFLINE] file_stem       = {file_stem}")
    print(f"[VIDEO/OFFLINE] out_pt_path     = {out_pt_path}")
    print(f"[VIDEO/OFFLINE] log_csv_path    = {log_csv_path}")
    print(f"[VIDEO/OFFLINE] target_clip_duration = {target_clip_duration}")

    if not timestamps_json_path.exists():
        print("[VIDEO/OFFLINE][ERROR] timestamps JSON does not exist — aborting.")
        return

    if not memsafe("before video offline save init"):
        return

    # 1) Build a temporary CSV just for this file
    temp_csv_path = out_pt_path.with_suffix(".words_tmp.csv")
    try:
        temp_csv_path = build_temp_csv_from_json(
            json_path=timestamps_json_path,
            file_stem=file_stem,
            temp_csv_path=temp_csv_path,
        )
    except Exception as e:
        print(f"[VIDEO/OFFLINE][ERROR] Failed to build temp CSV from JSON: {e}")
        return

    # 2) Run the final offline save helper (using the temp CSV)
    vcfg = VideoPreprocessorConfig()
    vp = VideoPreprocessorNPV(vcfg)

    num_segments, num_words = vp.process_and_save_from_timestamps_csv_segmentlocal(
        video_path=video_path,
        timestamps_csv_path=temp_csv_path,
        out_pt_path=out_pt_path,
        log_csv_path=log_csv_path,
        keep_full_when_no_face=True,
        min_factor=0.5,
        max_factor=1.5,
        target_clip_duration=target_clip_duration,
    )

    print(f"[VIDEO/OFFLINE] Saved {num_segments} segments from {num_words} words to {out_pt_path}")

    # 3) Inspect payload
    if not out_pt_path.exists():
        print("[VIDEO/OFFLINE][WARN] out_pt_path does not exist after save.")
        return

    payload = torch.load(out_pt_path)
    print(f"[VIDEO/OFFLINE] payload keys: {list(payload.keys())}")
    print(f"[VIDEO/OFFLINE] num_segments (payload): {payload.get('num_segments')}")
    print(f"[VIDEO/OFFLINE] num_words    (payload): {payload.get('num_words')}")

    video_segments = payload.get("video_segments")
    if video_segments and isinstance(video_segments, list):
        first = video_segments[0]
        if isinstance(first, torch.Tensor):
            print(f"[VIDEO/OFFLINE] first video segment shape: {tuple(first.shape)}")

    # Optional: clean up temp CSV
    # temp_csv_path.unlink(missing_ok=True)


# ====================================================================
# MAIN: SIMPLE MANUAL TEST
# ====================================================================

if __name__ == "__main__":
    # --------- Adjust these paths to match your dataset layout ---------
    SAMPLE_AUDIO = (
        "/Users/abhishekgupte_macbookpro/PycharmProjects/"
        "project_combined_repo_clean/thesis_main_files/data/processed/"
        "video_files/AVSpeech/audio/trim_audio_train100.wav"
    )
    SAMPLE_VIDEO = (
        "/Users/abhishekgupte_macbookpro/PycharmProjects/"
        "project_combined_repo_clean/thesis_main_files/data/processed/"
        "video_files/AVSpeech/video/video_83.mp4"
    )

    # Direct sanity uses simple word times (these can be synthetic):
    WORD_TIMES: List[List[float]] = [
        [0.20, 0.90],
        [1.00, 1.90],
    ]
    TARGET_CLIP_DURATION = 0.96  # seconds

    # Shared JSON with:
    #   { "trim_audio_train100": [[0.2, 0.9], [1.0, 1.9]],
    #     "video_83": [[0.2, 0.9], [1.0, 1.9]],
    #     ... }
    TS_JSON = (
        "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/AVSpeech/AVSpeech_timestamp_json_for_offline_training/AVSpeech_timestamp_json_for_offline_training.json"  # TODO: update this to real JSON path
    )

    # Where to dump offline .pt outputs for sanity:
    AUDIO_OUT_PT = (
        "/Users/abhishekgupte_macbookpro/PycharmProjects/"
        "project_combined_repo_clean/thesis_main_files/data/processed/"
        "video_files/AVSpeech/audio/sanity_trim_audio_train100_segments.pt"
    )
    VIDEO_OUT_PT = (
        "/Users/abhishekgupte_macbookpro/PycharmProjects/"
        "project_combined_repo_clean/thesis_main_files/data/processed/"
        "video_files/AVSpeech/video/sanity_video_83_segments.pt"
    )

    # Optional CSV logs:
    AUDIO_LOG_CSV = (
        "/Users/abhishekgupte_macbookpro/PycharmProjects/"
        "project_combined_repo_clean/thesis_main_files/data/processed/"
        "video_files/AVSpeech/audio/sanity_audio_log.csv"
    )
    VIDEO_LOG_CSV = (
        "/Users/abhishekgupte_macbookpro/PycharmProjects/"
        "project_combined_repo_clean/thesis_main_files/data/processed/"
        "video_files/AVSpeech/video/sanity_video_log.csv"
    )

    # ---------- Direct preprocess sanity (already working) ----------
    sanity_audio_preprocessor(
        audio_path=SAMPLE_AUDIO,
        word_times=WORD_TIMES,
    )

    sanity_video_preprocessor(
        video_path=SAMPLE_VIDEO,
        word_times=WORD_TIMES,
        target_clip_duration=TARGET_CLIP_DURATION,
    )

    # ---------- Offline-save sanity using the final helper methods ----------
    sanity_audio_offline_save(
        audio_path=SAMPLE_AUDIO,
        timestamps_json_path=TS_JSON,
        out_pt_path=AUDIO_OUT_PT,
        log_csv_path=AUDIO_LOG_CSV,
    )

    sanity_video_offline_save(
        video_path=SAMPLE_VIDEO,
        timestamps_json_path=TS_JSON,
        out_pt_path=VIDEO_OUT_PT,
        log_csv_path=VIDEO_LOG_CSV,
        target_clip_duration=TARGET_CLIP_DURATION,
    )
