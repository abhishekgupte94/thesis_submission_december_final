# audio_preprocessor_npv.py

"""
AudioPreprocessorNPV

Paper-safe, NPVForensics-style audio preprocessing class,
adapted for use inside PyTorch Lightning DataModules / DataLoaders.

It:
    - loads audio (file or tensor)
    - converts to mono
    - resamples to target_sr (e.g., 16 kHz)
    - peak-normalizes
    - computes log-mel spectrogram (n_mels, target_num_frames)
    - optionally normalizes per utterance

[DL-INTEGRATION]:
    Heavy torchaudio transforms are lazily created per worker and are
    excluded from pickling, so this class plays nicely with num_workers>0.

[TEMPORAL CLIPPING – OPTION B]:
    - We add a NPV-style, word-timestamp-driven temporal segmentation:
        * Given word-level timestamps [[t0_start, t0_end], [t1_start, t1_end], ...]
        * We group consecutive words into segments whose duration is
          ~ one Mel clip length (≈ target_num_frames * hop_length / target_sr),
          with allowable band [0.5x, 1.5x] of that duration.
        * For each segment, we slice the waveform and compute an
          individual (n_mels, target_num_frames) log-mel clip.

    - This produces a sequence of clips that mirrors NPVForensics'
      “segment fine-grained analysis” while respecting the paper’s
      Mel clip length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from einops import rearrange


class AudioPreprocessorNPV:
    """
    AudioPreprocessorNPV

    A configurable audio preprocessing class suitable for use in
    PyTorch Lightning DataModules / DataLoaders.

    Parameters
    ----------
    target_sr : int, default=16000
        Target sampling rate for all audio.

    n_fft : int, default=1024
        FFT size for STFT (via MelSpectrogram).

    win_length : int, default=400
        Window size in samples.

    hop_length : int, default=160
        Hop size in samples.

    n_mels : int, default=64
        Number of mel bands.

    f_min : float, default=50.0
        Minimum frequency for mel filters.

    f_max : float or None, default=None
        Maximum frequency for mel filters. If None, uses sr / 2.

    target_num_frames : int, default=96
        Target time dimension length. Spectrogram is cropped/padded along
        time to reach this length.

    normalize : bool, default=True
        If True, apply per-utterance mean/var normalization.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        n_fft: int = 1024,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
        f_min: float = 50.0,
        f_max: Optional[float] = None,
        target_num_frames: int = 96,
        normalize: bool = True,
    ) -> None:
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.target_num_frames = target_num_frames
        self.normalize = normalize

        # [DL-INTEGRATION]: heavy transforms are lazily created per worker.
        self._mel_transform: Optional[torchaudio.transforms.MelSpectrogram] = None
        self._db_transform: Optional[torchaudio.transforms.AmplitudeToDB] = None

    # ------------------------------------------------------------------
    # [DL-INTEGRATION]: ensure heavy objects are not pickled
    # ------------------------------------------------------------------
    def __getstate__(self):
        """Custom pickling: drop heavy torchaudio transforms."""
        state = self.__dict__.copy()
        state["_mel_transform"] = None
        state["_db_transform"] = None
        return state

    def __setstate__(self, state):
        """Re-initialize heavy transforms lazily in each worker."""
        self.__dict__.update(state)
        self._mel_transform = None
        self._db_transform = None

    # ------------------------------------------------------------------
    # [NEW - temporal clipping] Convenience: nominal Mel clip duration
    # ------------------------------------------------------------------
    @property
    def clip_duration_seconds(self) -> float:
        """
        Approximate duration (in seconds) of one Mel clip, based on the
        current STFT parameters.

        We mirror the NPVForensics setting where audio is resampled to
        16 kHz and converted into 96x64 log-mel spectrograms as input.
        """
        # Reasonable approximation: T frames * hop / sr
        return float(self.target_num_frames * self.hop_length) / float(self.target_sr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_transforms(self) -> None:
        if self._mel_transform is None:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_sr,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.n_mels,
                center=True,
                pad_mode="reflect",
                power=2.0,
                normalized=False,
            )
        if self._db_transform is None:
            self._db_transform = torchaudio.transforms.AmplitudeToDB(
                stype="power", top_db=80.0
            )

    @staticmethod
    def _load_audio_file(path: str) -> Tuple[Tensor, int]:
        wav, sr = torchaudio.load(path)  # (C, T)
        return wav, int(sr)

    def _standardize_waveform(self, wav: Tensor, sr: int) -> Tensor:
        """
        - Ensure mono
        - Resample to target_sr
        - Peak-normalize
        """
        if wav.ndim == 2:
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected waveform shape: {tuple(wav.shape)}")

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        # Peak-normalize per utterance
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak

        return wav  # (1, T_resampled)

    def _waveform_to_logmel(self, wav: Tensor) -> Tensor:
        """
        Convert mono waveform (1, T) @ target_sr to
        log-mel spectrogram of shape (n_mels, target_num_frames).
        """
        self._ensure_transforms()
        assert self._mel_transform is not None
        assert self._db_transform is not None

        mel = self._mel_transform(wav)  # (1, n_mels, T_frames)
        mel = self._db_transform(mel)   # (1, n_mels, T_frames)
        mel = mel.squeeze(0)            # (n_mels, T_frames)

        # Time dimension is last (T_frames)
        n_mels, T = mel.shape

        # Crop or pad along time to target_num_frames
        if T > self.target_num_frames:
            mel = mel[:, : self.target_num_frames]
        elif T < self.target_num_frames:
            pad_T = self.target_num_frames - T
            mel = F.pad(mel, (0, pad_T), mode="constant", value=0.0)

        if self.normalize:
            mean = mel.mean()
            std = mel.std(unbiased=False)
            if std > 0:
                mel = (mel - mean) / std

        return mel  # (n_mels, target_num_frames)

    # ------------------------------------------------------------------
    # Public “single-clip” APIs (backwards compatible)
    # ------------------------------------------------------------------
    def process_audio_file(self, path: str) -> Tensor:
        """
        Legacy single-clip API: produce ONE log-mel clip for the whole audio.
        """
        wav, sr = self._load_audio_file(path)
        wav = self._standardize_waveform(wav, sr)
        mel = self._waveform_to_logmel(wav)
        return mel

    def __call__(self, path: str) -> Tensor:
        return self.process_audio_file(path)

    # ------------------------------------------------------------------
    # [NEW - temporal clipping] Word-timestamp driven segmentation
    # ------------------------------------------------------------------
    @staticmethod
    def build_segments_from_word_times(
        word_times: Sequence[Sequence[float]],
        target_clip_duration: float,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> List[Tuple[float, float]]:
        """
        Construct temporal segments by grouping consecutive words so that
        each segment duration is approx. the Mel clip duration.

        Parameters
        ----------
        word_times:
            Sequence like [[start_0, end_0], [start_1, end_1], ...].
            These correspond to word-level timestamps already present
            in your dataset.

        target_clip_duration:
            Desired segment duration in seconds, derived from the audio
            preprocessing hyperparameters (≈ 0.96 s in NPVForensics).

        min_factor, max_factor:
            Segment durations are kept within:
                [min_factor * target_clip_duration,
                 max_factor * target_clip_duration]

        Returns
        -------
        segments:
            List of (seg_start_sec, seg_end_sec).
        """
        if not word_times:
            return []

        # Ensure proper ordering
        word_times_sorted = sorted(word_times, key=lambda x: x[0])

        min_dur = min_factor * target_clip_duration
        max_dur = max_factor * target_clip_duration

        segments: List[Tuple[float, float]] = []
        i = 0
        n = len(word_times_sorted)

        while i < n:
            seg_start = float(word_times_sorted[i][0])
            seg_end = float(word_times_sorted[i][1])
            j = i + 1

            # Grow the segment by adding consecutive words until:
            #   - we are close to target_clip_duration, or
            #   - we hit max_dur, or
            #   - we run out of words.
            while j < n:
                candidate_end = float(word_times_sorted[j][1])
                candidate_dur = candidate_end - seg_start

                if candidate_dur > max_dur:
                    break

                seg_end = candidate_end
                j += 1

                # If we are at or beyond the target duration, we are happy
                if seg_end - seg_start >= target_clip_duration:
                    break

            seg_dur = seg_end - seg_start

            # If the resulting segment is still very short, we can either:
            #   - merge it with the next (if exists), or
            #   - keep it as is for edge cases.
            if seg_dur < min_dur and j < n:
                # Try merging one more word if possible
                candidate_end = float(word_times_sorted[j][1])
                candidate_dur = candidate_end - seg_start
                if candidate_dur <= max_dur:
                    seg_end = candidate_end
                    j += 1
                    seg_dur = seg_end - seg_start

            # Guard against degenerate segments
            if seg_end > seg_start:
                segments.append((seg_start, seg_end))

            i = max(j, i + 1)

        return segments

    def _slice_waveform_by_segments(
        self,
        wav: Tensor,
        sr: int,
        segments: Sequence[Tuple[float, float]],
    ) -> List[Tensor]:
        """
        Slice a mono waveform (1, T) into a list of segments using
        absolute time boundaries in seconds.
        """
        if wav.ndim != 2 or wav.size(0) != 1:
            raise ValueError(
                f"Expected mono waveform of shape (1, T), got {tuple(wav.shape)}"
            )

        T = wav.size(1)
        clips: List[Tensor] = []

        for (start_sec, end_sec) in segments:
            start_sample = int(round(start_sec * sr))
            end_sample = int(round(end_sec * sr))
            start_sample = max(0, min(start_sample, T - 1))
            end_sample = max(start_sample + 1, min(end_sample, T))

            clip = wav[:, start_sample:end_sample]  # (1, T_seg)
            clips.append(clip)

        return clips

    def process_file_with_word_segments(
        self,
        path: str,
        word_times: Sequence[Sequence[float]],
        min_factor: float = 0.5,
        max_factor: float = 1.5,
        return_segments: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tuple[float, float]]]]:
        """
        NEW NPV-style multi-clip API:

        Given:
            - an audio file path
            - a list of word timestamps [[s0, e0], [s1, e1], ...] in seconds

        1) Build VA-consistent temporal segments using word grouping
           that respects the paper's Mel clip duration.
        2) Slice the waveform by these segments.
        3) Convert each slice to a (n_mels, target_num_frames) log-mel clip.

        Returns
        -------
        mel_clips:
            Tensor of shape (N_segments, n_mels, target_num_frames)

        segments (optional, if return_segments=True):
            List of (start_sec, end_sec) used for alignment with frames.
        """
        # 1) Build temporal segments (Option B, time-aware)
        target_clip_dur = self.clip_duration_seconds
        segments = self.build_segments_from_word_times(
            word_times=word_times,
            target_clip_duration=target_clip_dur,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        # Fallback: if no segments could be built, use the full utterance
        if not segments:
            wav, sr = self._load_audio_file(path)
            wav = self._standardize_waveform(wav, sr)
            mel = self._waveform_to_logmel(wav)
            mel = mel.unsqueeze(0)  # (1, n_mels, target_num_frames)
            return (mel, None if not return_segments else [(0.0, float(wav.size(1) / sr))])

        # 2) Load waveform once, standardize
        wav, sr = self._load_audio_file(path)
        wav = self._standardize_waveform(wav, sr)

        # 3) Slice into segments and compute log-mel per slice
        wav_clips = self._slice_waveform_by_segments(wav, self.target_sr, segments)
        mel_clips: List[Tensor] = []
        for clip_wav in wav_clips:
            mel = self._waveform_to_logmel(clip_wav)
            mel_clips.append(mel)

        mel_stack = torch.stack(mel_clips, dim=0)  # (N_segments, n_mels, T)

        return mel_stack, (segments if return_segments else None)
