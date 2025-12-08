# unified_npv_preprocessing_and_tokenisation.py
"""
[ADDED] Unified wrapper that connects:
    - AudioPreprocessorNPV (waveform -> NPV segments of log-mel clips)
    - VideoPreprocessorNPV (video -> face crops per frame)
    - MultiModalTokeniser   (per-modality projection + positional embeddings)

The goal is to give you a single call that takes:
    (audio_path, video_path, word_times, video_id, frame roots)
and returns *tokenised & position-embedded* audio / frame sequences that are
ready to be fed into your downstream Swin / LFA-ST / VACL blocks.

This file does NOT depend on any CLI / argparse logic and is safe to import
inside Lightning DataModules, Datasets, or offline preprocessing scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from AudioPreprocessorNPV import AudioPreprocessorNPV   # [ASSUME] same package / PYTHONPATH
from VideoPreprocessorNPV import VideoPreprocessorNPV   # [ASSUME] same package / PYTHONPATH
from positional_embedding_tokenization import MultiModalTokeniser


# -------------------------------------------------------------------------
# [ADDED] Small config helper for this unified wrapper
# -------------------------------------------------------------------------
@dataclass
class UnifiedNPVConfig:
    """
    [EXPLAIN] Configuration for the unified audio+video preprocessing + tokenisation
    pipeline.

    This "sits above" the individual modality configs in positional_embedding_tokenization.py.

    Fields
    ------
    frame_image_size:
        (H, W) to which cropped faces will be resized *before* flattening.
        MUST be consistent with the frame_input_dim you passed to
        ModalityConfig(input_dim=3 * H * W, ...) for the "frame" modality.

    max_frame_tokens:
        Maximum Sv (number of frames per segment) that we keep for the frame
        modality. If a segment has more frames, we truncate; if fewer, we pad
        with zeros so every segment has a fixed Sv.

        MUST be <= cfg_frame.max_seq_len used when you built the frame tokeniser.

    device:
        Where to place the constructed tensors (e.g. "cpu" or "cuda:0").
        This only controls the token input tensors, not the internal torchaudio /
        OpenCV operations.
    """

    frame_image_size: Tuple[int, int]  # (H, W)
    max_frame_tokens: int
    device: str = "cpu"


# -------------------------------------------------------------------------
# [ADDED] Core wrapper class
# -------------------------------------------------------------------------
class UnifiedNPVPreprocessAndTokenise:
    """
    [EXPLAIN] High-level object that wires together:

        AudioPreprocessorNPV
        VideoPreprocessorNPV
        MultiModalTokeniser (with "audio" and "frame" entries)

    Typical usage
    -------------
    audio_prep = AudioPreprocessorNPV(...)
    video_prep = VideoPreprocessorNPV(...)
    tokeniser  = build_frame_audio_tokeniser(...)

    cfg = UnifiedNPVConfig(
        frame_image_size=(112, 112),
        max_frame_tokens=32,
        device="cuda:0",
    )

    wrapper = UnifiedNPVPreprocessAndTokenise(
        audio_preprocessor=audio_prep,
        video_preprocessor=video_prep,
        tokeniser=tokeniser,
        cfg=cfg,
    )

    out = wrapper.process_sample(
        audio_path=".../audio.wav",
        video_path=".../video.mp4",
        word_times=[[0.10, 0.35], [0.36, 0.72], ...],
        video_id="sample_001",
        frames_root="/path/to/raw_frames_root",
        cropped_root="/path/to/cropped_frames_root",
    )

    audio_tokens = out["audio_tokens"]  # (N_segments, Sa, d_model)
    frame_tokens = out["frame_tokens"]  # (N_segments, Sv, d_model)
    segments     = out["segments"]      # list of (start_sec, end_sec)
    """

    def __init__(
        self,
        audio_preprocessor: AudioPreprocessorNPV,
        video_preprocessor: VideoPreprocessorNPV,
        tokeniser: MultiModalTokeniser,
        cfg: UnifiedNPVConfig,
    ) -> None:
        super().__init__()

        self.audio_preprocessor = audio_preprocessor
        self.video_preprocessor = video_preprocessor
        self.tokeniser = tokeniser
        self.cfg = cfg

        # [SAFEGUARD] Check that required modalities exist in the tokeniser
        expected_modalities = {"audio", "frame"}
        missing = expected_modalities.difference(set(self.tokeniser.tokenisers.keys()))
        if missing:
            raise KeyError(
                f"[UnifiedNPVPreprocessAndTokenise] Tokeniser is missing modalities: "
                f"{missing}. Available: {list(self.tokeniser.tokenisers.keys())}"
            )

        # [CACHE] Convenience accessors for per-modality tokenisers
        self._audio_tok = self.tokeniser.tokenisers["audio"]
        self._frame_tok = self.tokeniser.tokenisers["frame"]

        # [SAFEGUARD] Frame input dim must match 3 * H * W
        H, W = self.cfg.frame_image_size
        expected_frame_dim = 3 * H * W
        if self._frame_tok.cfg.input_dim != expected_frame_dim:
            raise ValueError(
                "[UnifiedNPVPreprocessAndTokenise] Frame tokeniser cfg.input_dim "
                f"= {self._frame_tok.cfg.input_dim}, but expected 3*H*W = "
                f"{expected_frame_dim} for H={H}, W={W}. Make sure you built the "
                "frame ModalityConfig with input_dim=3*H*W matching frame_image_size."
            )

        # [SAFEGUARD] max_frame_tokens must not exceed frame.cfg.max_seq_len
        if self.cfg.max_frame_tokens > self._frame_tok.cfg.max_seq_len:
            raise ValueError(
                "[UnifiedNPVPreprocessAndTokenise] cfg.max_frame_tokens "
                f"({self.cfg.max_frame_tokens}) exceeds frame tokeniser "
                f"max_seq_len ({self._frame_tok.cfg.max_seq_len})."
            )

    # ------------------------------------------------------------------
    # [ADDED] Public API: process a single paired audio/video sample
    # ------------------------------------------------------------------
    def process_sample(
        self,
        audio_path: str,
        video_path: str,
        word_times: Sequence[Sequence[float]],
        video_id: Optional[str],
        frames_root: str,
        cropped_root: str,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
        return_segments: bool = True,
    ) -> Dict[str, object]:
        """
        [EXPLAIN] End-to-end pipeline for *one* sample.

        1) Use AudioPreprocessorNPV to:
            - Build NPV-style temporal segments from word_times.
            - Slice the waveform accordingly.
            - Convert each slice to a log-mel clip.
            - Stack into (N_segments, n_mels, T_mel).

        2) Use VideoPreprocessorNPV to:
            - Extract frames and crop faces to disk.
            - Track FPS and total frame count.
            - Map the same temporal segments to frame index ranges.
            - Group cropped frame paths into per-segment lists.

        3) Convert:
            - audio mel stack   -> (N_segments, Sa, D_audio_in) for tokeniser
            - frame crops stack -> (N_segments, Sv, D_frame_in) for tokeniser

        4) Apply the modality-specific tokenisers (projection + positional
           embeddings) to obtain:
            audio_tokens: (N_segments, Sa, d_model)
            frame_tokens: (N_segments, Sv, d_model)

        Returns
        -------
        dict with keys:
            "audio_tokens": Tensor
            "frame_tokens": Tensor
            "segments":     List[(start_sec, end_sec)]
            "frame_paths":  List[List[Path]]  # per segment, for debugging/inspection
        """

        # ------------------------
        # 1) Audio: mel segments
        # ------------------------
        mel_stack, segments = self.audio_preprocessor.process_file_with_word_segments(
            path=audio_path,
            word_times=word_times,
            min_factor=min_factor,
            max_factor=max_factor,
            return_segments=True,
        )
        # mel_stack: (N_segments, n_mels, T_mel)
        if mel_stack.ndim != 3:
            raise RuntimeError(
                f"[Unified] Expected mel_stack of shape (N_segments, n_mels, T), "
                f"got {tuple(mel_stack.shape)}"
            )

        N_segments, n_mels, T_mel = mel_stack.shape

        if N_segments == 0:
            raise RuntimeError(
                "[UnifiedNPVPreprocessAndTokenise] No audio segments were produced. "
                "Check your word_times and audio file."
            )

        # [EXPLAIN] Reorder to (B, S, D_in) for the audio tokeniser:
        #   B = N_segments, S = Sa = T_mel, D_in = n_mels
        audio_features = mel_stack.permute(0, 2, 1)  # (N_segments, T_mel, n_mels)

        # ------------------------
        # 2) Video: frame crops
        # ------------------------
        raw_frame_paths, cropped_paths = self.video_preprocessor.process_video_file(
            video_path=video_path,
            frames_root=frames_root,
            cropped_root=cropped_root,
            video_id=video_id,
            keep_full_when_no_face=True,
        )
        num_frames = len(cropped_paths)
        if num_frames == 0:
            raise RuntimeError(
                "[UnifiedNPVPreprocessAndTokenise] No cropped frames were produced "
                f"for video {video_path}."
            )

        fps = float(self.video_preprocessor._last_fps)
        if fps <= 0:
            raise RuntimeError(
                "[UnifiedNPVPreprocessAndTokenise] VideoPreprocessorNPV._last_fps "
                f"is {fps}. Make sure extract_frames() was called successfully."
            )

        # [EXPLAIN] Map the *audio-defined* temporal segments to frame ranges.
        from VideoPreprocessorNPV import (
            segments_to_frame_indices,
            group_frame_paths_by_segments,
        )

        frame_ranges = segments_to_frame_indices(
            segments=segments,
            fps=fps,
            num_frames=num_frames,
        )  # List[(start_idx, end_idx)]

        per_segment_frame_paths = group_frame_paths_by_segments(
            frame_paths=cropped_paths,
            frame_ranges=frame_ranges,
        )  # List[List[Path]]

        # ------------------------
        # 3) Tensorise video segments for tokeniser
        # ------------------------
        frame_features = self._build_frame_features_tensor(per_segment_frame_paths)
        # shape: (N_segments, Sv, D_frame_in)

        # [SAFEGUARD] N_segments alignment
        if frame_features.size(0) != N_segments:
            raise RuntimeError(
                "[UnifiedNPVPreprocessAndTokenise] Mismatch between "
                f"audio segments (N={N_segments}) and frame segment groups "
                f"(N={frame_features.size(0)}). Check your segment logic / timings."
            )

        # ------------------------
        # 4) Apply tokenisers (projection + positional embeddings)
        # ------------------------
        device = torch.device(self.cfg.device)
        audio_features = audio_features.to(device)
        frame_features = frame_features.to(device)

        audio_tokens = self._audio_tok(audio_features)  # (N_segments, Sa, d_model)
        frame_tokens = self._frame_tok(frame_features)  # (N_segments, Sv, d_model)

        return {
            "audio_tokens": audio_tokens,
            "frame_tokens": frame_tokens,
            "segments": segments if return_segments else None,
            "frame_paths": per_segment_frame_paths,
        }

    # ------------------------------------------------------------------
    # [ADDED] Helper: convert per-segment frame paths to (N, Sv, D_frame_in)
    # ------------------------------------------------------------------
    def _build_frame_features_tensor(
        self,
        per_segment_frame_paths: Sequence[Sequence[Path]],
    ) -> Tensor:
        """
        [EXPLAIN] Load cropped face frames from disk, resize, flatten, and
        pack into a 3D tensor suitable for the "frame" ModalityTokeniser.

        - Each segment i can have a variable number of frames.
        - We clamp each to at most cfg.max_frame_tokens (truncate if longer).
        - If a segment has fewer frames, we pad with zeros.

        Output shape
        ------------
        frame_features : Tensor
            Shape (N_segments, Sv, D_frame_in), where:
                N_segments  = len(per_segment_frame_paths)
                Sv          = cfg.max_frame_tokens
                D_frame_in  = 3 * H * W
        """
        H, W = self.cfg.frame_image_size
        Sv = self.cfg.max_frame_tokens
        D_frame_in = 3 * H * W

        segment_tensors: List[Tensor] = []

        for seg_idx, frame_paths in enumerate(per_segment_frame_paths):
            # [EXPLAIN] Load each frame, resize, convert BGR->RGB, flatten
            frames_vecs: List[Tensor] = []
            for p in frame_paths:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                # Convert to float32 in [0,1] and flatten
                img_t = torch.from_numpy(img).float() / 255.0  # (H, W, 3)
                img_t = img_t.permute(2, 0, 1)  # (3, H, W)
                frames_vecs.append(img_t.reshape(-1))  # (D_frame_in,)

            if len(frames_vecs) == 0:
                # [EXPLAIN] If this segment ended up with no usable frames,
                # create a single zero vector to avoid crashing.
                frames_mat = torch.zeros(1, D_frame_in, dtype=torch.float32)
            else:
                frames_mat = torch.stack(frames_vecs, dim=0)  # (num_frames_seg, D_frame_in)

            num_frames_seg = frames_mat.size(0)

            # [EXPLAIN] Truncate or pad to Sv
            if num_frames_seg > Sv:
                frames_trimmed = frames_mat[:Sv, :]
            elif num_frames_seg < Sv:
                pad_rows = Sv - num_frames_seg
                pad = torch.zeros(pad_rows, D_frame_in, dtype=torch.float32)
                frames_trimmed = torch.cat([frames_mat, pad], dim=0)
            else:
                frames_trimmed = frames_mat

            segment_tensors.append(frames_trimmed)  # (Sv, D_frame_in)

        # Stack segments into batch dimension
        frame_features = torch.stack(segment_tensors, dim=0)  # (N_segments, Sv, D_frame_in)
        return frame_features
