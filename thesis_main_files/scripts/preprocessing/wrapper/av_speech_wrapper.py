# AVPreprocessorNPV.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

# Import your existing preprocessors (adjust import paths to match your repo)
from scripts.preprocessing.audio.AudioPreprocessorNPV import (
    AudioPreprocessorNPV,
    AudioPreprocessorConfig,
)
from scripts.preprocessing.video.VideoPreprocessorNPV import (
    VideoPreprocessorNPV,
    VideoPreprocessorConfig,
)


PathLike = Union[str, Path]


# ============================================================
# [NEW] Unified AV config
# ============================================================
@dataclass
class AVPreprocessorConfig:
    audio: AudioPreprocessorConfig
    video: VideoPreprocessorConfig

    # Naming convention for saved artifacts
    audio_suffix: str = "audio"
    video_suffix: str = "video"

    # Optional: create output dirs if missing
    mkdirs: bool = True


# ============================================================
# [NEW] Unified AV wrapper
# ============================================================
class AVPreprocessorNPV:
    """
    Thin-but-useful unified wrapper around:
      - AudioPreprocessorNPV
      - VideoPreprocessorNPV

    Goal: one entrypoint for your offline exporter / sanity scripts / dataloader prep.
    """

    def __init__(self, cfg: AVPreprocessorConfig):
        self.cfg = cfg
        self.audio = AudioPreprocessorNPV(cfg.audio)
        self.video = VideoPreprocessorNPV(cfg.video)

    # ----------------------------
    # Path helpers
    # ----------------------------
    def _ensure_dir(self, p: Path) -> None:
        if self.cfg.mkdirs:
            p.mkdir(parents=True, exist_ok=True)

    def build_out_paths(
        self,
        out_dir: PathLike,
        clip_id: str,
        *,
        audio_ext: str = ".pt",
        video_ext: str = ".pt",
        audio_log_ext: str = ".csv",
    ) -> Dict[str, Path]:
        """
        Standardises how artifacts are named and where they go.

        Produces:
          - <out_dir>/<clip_id>_<audio_suffix>.pt
          - <out_dir>/<clip_id>_<video_suffix>.pt
          - <out_dir>/<clip_id>_<audio_suffix>_log.csv
        """
        out_dir = Path(out_dir)
        self._ensure_dir(out_dir)

        audio_pt = out_dir / f"{clip_id}_{self.cfg.audio_suffix}{audio_ext}"
        video_pt = out_dir / f"{clip_id}_{self.cfg.video_suffix}{video_ext}"
        audio_log = out_dir / f"{clip_id}_{self.cfg.audio_suffix}_log{audio_log_ext}"

        return {
            "audio_pt": audio_pt,
            "video_pt": video_pt,
            "audio_log": audio_log,
        }

    # ----------------------------
    # In-memory processing
    # ----------------------------
    def process_pair_with_word_segments_segmentlocal(
        self,
        *,
        audio_path: PathLike,
        video_path: PathLike,
        word_segments: Any,
        # pass-through knobs (keep defaults identical to underlying preprocessors)
        keep_full_when_no_face: bool = True,
        min_factor: float = 0.75,
        max_factor: float = 1.25,
        target_clip_duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Runs BOTH preprocessors in-memory, given a pre-built `word_segments`
        structure (whatever your preprocessors already expect).

        Returns a dict:
          {
            "audio": <audio_segments>,
            "video": <video_segments>,
          }
        """
        audio_path = Path(audio_path)
        video_path = Path(video_path)

        # Audio: segment-local pathway (your existing method)
        audio_out = self.audio.process_file_with_word_segments_segmentlocal(
            audio_path=str(audio_path),
            word_segments=word_segments,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        # Video: tensor/segment-local pathway (your existing method)
        # NOTE: your video preprocessor has both list and tensor variants;
        # we use the tensor variant because that’s typically what goes to Swin.
        video_out = self.video.process_video_file_with_word_segments_tensor(
            video_path=str(video_path),
            word_segments=word_segments,
            keep_full_when_no_face=keep_full_when_no_face,
            min_factor=min_factor,
            max_factor=max_factor,
            target_clip_duration=target_clip_duration,
        )

        return {"audio": audio_out, "video": video_out}

    # ----------------------------
    # CSV-driven processing + saving
    # ----------------------------
    def process_and_save_pair_from_timestamps_csv_segmentlocal(
        self,
        *,
        audio_path: PathLike,
        video_path: PathLike,
        timestamps_csv_path: PathLike,
        out_dir: PathLike,
        clip_id: str,
        # pass-through knobs
        keep_full_when_no_face: bool = True,
        min_factor: float = 0.75,
        max_factor: float = 1.25,
        target_clip_duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Single call that:
          1) Runs audio CSV-driven segment-local saver
          2) Runs video CSV-driven segment-local saver
          3) Uses consistent filenames via `clip_id`

        Returns:
          {
            "audio": {"out_pt": Path, "log_csv": Path, "num_segments": int, "num_words": int},
            "video": {"out_pt": Path, "num_segments": int, "num_words": int},
          }
        """
        audio_path = Path(audio_path)
        video_path = Path(video_path)
        timestamps_csv_path = Path(timestamps_csv_path)

        outs = self.build_out_paths(out_dir=out_dir, clip_id=clip_id)

        # [AUDIO] uses a log CSV in your implementation
        a_num_segments, a_num_words = self.audio.process_and_save_from_timestamps_csv_segmentlocal(
            audio_path=str(audio_path),
            timestamps_csv_path=str(timestamps_csv_path),
            out_pt_path=str(outs["audio_pt"]),
            log_csv_path=str(outs["audio_log"]),
            min_factor=min_factor,
            max_factor=max_factor,
        )

        # [VIDEO] no log CSV in your current signature
        v_num_segments, v_num_words = self.video.process_and_save_from_timestamps_csv_segmentlocal(
            video_path=str(video_path),
            timestamps_csv_path=str(timestamps_csv_path),
            out_pt_path=str(outs["video_pt"]),
            keep_full_when_no_face=keep_full_when_no_face,
            min_factor=min_factor,
            max_factor=max_factor,
            target_clip_duration=target_clip_duration,
        )

        return {
            "audio": {
                "out_pt": outs["audio_pt"],
                "log_csv": outs["audio_log"],
                "num_segments": a_num_segments,
                "num_words": a_num_words,
            },
            "video": {
                "out_pt": outs["video_pt"],
                "num_segments": v_num_segments,
                "num_words": v_num_words,
            },
        }

    # ----------------------------
    # Convenience: build from defaults quickly
    # ----------------------------
    @staticmethod
    def build_default(
        *,
        audio_cfg: Optional[AudioPreprocessorConfig] = None,
        video_cfg: Optional[VideoPreprocessorConfig] = None,
    ) -> "AVPreprocessorNPV":
        """
        Convenience ctor for quick sanity scripts.
        """
        if audio_cfg is None:
            audio_cfg = AudioPreprocessorConfig()
        if video_cfg is None:
            video_cfg = VideoPreprocessorConfig()

        return AVPreprocessorNPV(AVPreprocessorConfig(audio=audio_cfg, video=video_cfg))
