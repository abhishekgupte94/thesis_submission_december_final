# audio_preprocessor_npv.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import csv
import torch
import torchaudio
from torch import Tensor


def _get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "thesis_main_files":
            return parent
    return here.parents[3]


def _to_rel_data_path(path: Path) -> str:
    project_root = _get_project_root()
    try:
        rel = path.resolve().relative_to(project_root)
    except Exception:
        rel = path.resolve()
    return rel.as_posix()


def build_segments_from_word_times(
    word_times: Sequence[Sequence[float]],
    *,
    target_clip_duration: float,
    min_factor: float = 0.5,
    max_factor: float = 1.5,
) -> List[Tuple[float, float]]:
    """
    Convert word time-stamps into segment windows.
    """
    if target_clip_duration <= 0:
        raise ValueError("target_clip_duration must be > 0")

    segments: List[Tuple[float, float]] = []
    for w in word_times:
        if len(w) < 3:
            continue
        _, start, end = w[0], float(w[1]), float(w[2])
        if end < start:
            continue

        dur = end - start
        # Clamp duration based on factors
        lo = dur * min_factor
        hi = dur * max_factor
        seg_dur = max(lo, min(hi, target_clip_duration))

        seg_start = start
        seg_end = min(seg_start + seg_dur, end if end > seg_start else seg_start + seg_dur)
        if seg_end <= seg_start:
            continue
        segments.append((seg_start, seg_end))
    return segments


@dataclass
class AudioPreprocessorNPVConfig:
    target_sr: int = 16000
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    f_min: float = 0.0
    f_max: Optional[float] = None
    eps: float = 1e-10
    target_num_frames: int = 96
    normalize_utterance: bool = True


class AudioPreprocessorNPV:
    def __init__(self, cfg: AudioPreprocessorNPVConfig) -> None:
        self.cfg = cfg
        self.target_sr = int(cfg.target_sr)
        self.n_mels = int(cfg.n_mels)
        self.n_fft = int(cfg.n_fft)
        self.hop_length = int(cfg.hop_length)
        self.win_length = int(cfg.win_length)
        self.f_min = float(cfg.f_min)
        self.f_max = cfg.f_max
        self.eps = float(cfg.eps)
        self.target_num_frames = int(cfg.target_num_frames)
        self.normalize_utterance = bool(cfg.normalize_utterance)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )

    def _load_audio_file(self, path: Union[str, Path]) -> Tuple[Tensor, int]:
        path = Path(path)
        wav, sr = torchaudio.load(str(path))
        # Convert to mono if needed
        if wav.ndim == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.ndim == 2:
            wav = wav.squeeze(0)
        return wav, int(sr)

    def _standardize_waveform(self, wav: Tensor, sr: int) -> Tensor:
        # Resample if needed
        if int(sr) != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        # Normalize peak
        if self.normalize_utterance:
            peak = wav.abs().max()
            if peak > 0:
                wav = wav / peak
        return wav

    def _waveform_to_logmel(self, wav: Tensor) -> Tensor:
        mel = self.mel_spectrogram(wav)
        mel = torch.clamp(mel, min=self.eps)
        mel = torch.log(mel)

        T = mel.size(1)
        target_T = self.target_num_frames

        if T == target_T:
            return mel
        if T > target_T:
            start = (T - target_T) // 2
            mel = mel[:, start : start + target_T]
        else:
            pad_T = target_T - T
            pad = torch.zeros(self.n_mels, pad_T, dtype=mel.dtype)
            mel = torch.cat([mel, pad], dim=1)
        return mel


    # ============================================================
    # [ADDED] 64x2048 log-Mel creation (paper-style long context)
    #
    # CRITICAL REQUIREMENTS (per your instructions):
    #   - DO NOT change any existing wiring/logic.
    #   - Follow the EXACT same methodology as _waveform_to_logmel()
    #     (mel_spectrogram -> clamp -> log -> center-crop or zero-pad).
    #   - Only difference is target_T = 2048.
    # ============================================================
    def _waveform_to_logmel_2048(self, wav: Tensor) -> Tensor:
        mel = self.mel_spectrogram(wav)
        mel = torch.clamp(mel, min=self.eps)
        mel = torch.log(mel)

        T = mel.size(1)
        target_T = 2048  # [ADDED] fixed long-context time dimension

        if T == target_T:
            return mel
        if T > target_T:
            start = (T - target_T) // 2
            mel = mel[:, start : start + target_T]
        else:
            pad_T = target_T - T
            pad = torch.zeros(self.n_mels, pad_T, dtype=mel.dtype)
            mel = torch.cat([mel, pad], dim=1)
        return mel

    def process_audio_file(self, path: Union[str, Path]) -> Tensor:
        wav, sr = self._load_audio_file(path)
        wav = self._standardize_waveform(wav, sr)
        return self._waveform_to_logmel(wav)

    def slice_waveform_with_segments(
        self,
        wav: Tensor,
        segments_sec: Sequence[Tuple[float, float]],
    ) -> List[Tensor]:
        clips: List[Tensor] = []
        for start_sec, end_sec in segments_sec:
            start_idx = int(start_sec * self.target_sr)
            end_idx = int(end_sec * self.target_sr)
            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, wav.size(0))
            if end_idx <= start_idx:
                continue
            clip = wav[start_idx:end_idx]
            if self.normalize_utterance:
                peak = clip.abs().max()
                if peak > 0:
                    clip = clip / peak
            clips.append(clip)
        return clips

    def process_file_with_word_segments(
        self,
        path: Union[str, Path],
        word_times: Sequence[Sequence[float]],
        target_clip_duration: float,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> Tuple[List[Tensor], List[Tuple[float, float]]]:
        wav, sr = self._load_audio_file(path)
        wav = self._standardize_waveform(wav, sr)

        segments = build_segments_from_word_times(
            word_times,
            target_clip_duration=target_clip_duration,
            min_factor=min_factor,
            max_factor=max_factor,
        )
        clips = self.slice_waveform_with_segments(wav, segments)
        mel_segments: List[Tensor] = [self._waveform_to_logmel(c) for c in clips]
        return mel_segments, segments

    def process_file_with_word_segments_segmentlocal(
        self,
        path: Union[str, Path],
        word_times: Sequence[Sequence[float]],
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> Tuple[List[Tensor], List[Tuple[float, float]]]:
        target_clip_duration = float(self.target_num_frames) * float(self.hop_length) / float(self.target_sr)
        return self.process_file_with_word_segments(
            path=path,
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            min_factor=min_factor,
            max_factor=max_factor,
        )

    def process_and_save_from_segments_sec_segmentlocal(
            self,
            audio_path: Union[str, Path],
            segments_sec: Sequence[Tuple[float, float]],
            out_pt_path: Union[str, Path],
            log_csv_path: Optional[Union[str, Path]] = None,
            *,
            clip_id: Optional[Union[str, int]] = None,
            overwrite: bool = False,
    ) -> Tuple[int, int]:
        """
        [MODIFIED] Save EACH mel segment as its OWN .pt file directly under <clip_id>/.

        Treat out_pt_path as an OUTPUT ROOT DIRECTORY (not a file):
            out_root/<clip_id>/<clip_id>_0000.pt
            out_root/<clip_id>/<clip_id>_0001.pt
            ...

        Each .pt contains ONLY the mel Tensor (64,96).
        """
        audio_path = Path(audio_path)
        out_root = Path(out_pt_path)

        num_words = 0  # kept for caller compatibility

        clip_id_str = str(clip_id) if clip_id is not None else audio_path.stem

        wav, sr = self._load_audio_file(audio_path)
        wav = self._standardize_waveform(wav, sr)

        # ==========================================================
        # This is the exact timestamp-driven loop you referenced:
        # segments_sec comes from video .pt, so this aligns with video.
        # ==========================================================
        clips = self.slice_waveform_with_segments(wav, segments_sec)
        mel_segments: List[Tensor] = [self._waveform_to_logmel(c) for c in clips]
        # ==========================================================
        # [ADDED] Long-context mel (64x2048) computed from the SAME clips
        # Reason: preserve exact segmentation + align perfectly with video.
        # ==========================================================
        mel_segments_2048: List[Tensor] = [self._waveform_to_logmel_2048(c) for c in clips]
        num_segments = len(mel_segments)

        # dtype/shape safety (unchanged)
        for idx, mel in enumerate(mel_segments):
            if not isinstance(mel, torch.Tensor):
                raise TypeError(f"mel_segments[{idx}] is not a Tensor (got {type(mel)})")
            if mel.ndim != 2:
                raise ValueError(f"Expected mel_segments[{idx}] shape (n_mels, T), got {tuple(mel.shape)}")
            if mel.shape[0] != self.n_mels:
                raise ValueError(f"mel_segments[{idx}].shape[0]={mel.shape[0]} expected {self.n_mels}")
            if mel.dtype != torch.float32:
                mel_segments[idx] = mel.float()

        # ==========================================================
        # [ADDED] dtype/shape safety for 64x2048 tensors (fail fast)
        # NOTE: This does NOT alter the existing 64x96 checks above.
        # ==========================================================
        for idx, mel in enumerate(mel_segments_2048):
            if not isinstance(mel, torch.Tensor):
                raise TypeError(f"mel_segments_2048[{idx}] is not a Tensor (got {type(mel)})")
            if mel.ndim != 2:
                raise ValueError(f"Expected mel_segments_2048[{idx}] shape (n_mels, T), got {tuple(mel.shape)}")
            if mel.shape[0] != self.n_mels:
                raise ValueError(
                    f"mel_segments_2048[{idx}].shape[0]={mel.shape[0]} expected {self.n_mels}"
                )
            if mel.shape[1] != 2048:
                raise ValueError(
                    f"Expected mel_segments_2048[{idx}] time dim 2048, got {mel.shape[1]}"
                )
            if mel.dtype != torch.float32:
                mel_segments_2048[idx] = mel.float()

        # [MODIFIED] Directory: out_root/<clip_id>/
        clip_dir = out_root / clip_id_str
        clip_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for seg_idx, mel in enumerate(mel_segments):
            out_seg_pt = clip_dir / f"{clip_id_str}_{seg_idx:04d}.pt"

            if out_seg_pt.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing file: {out_seg_pt}")

            torch.save(mel, out_seg_pt)  # <-- payload is ONLY the Tensor

            # ==========================================================
            # [ADDED] Save the paired 64x2048 mel tensor next to the 64x96.
            # Naming: <clip_id>_<seg>__2048.pt (keeps original intact).
            # ==========================================================
            out_seg_pt_2048 = clip_dir / f"{clip_id_str}_{seg_idx:04d}__2048.pt"
            if out_seg_pt_2048.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing file: {out_seg_pt_2048}")
            torch.save(mel_segments_2048[seg_idx], out_seg_pt_2048)

            saved += 1

        # Logging
        if log_csv_path is not None:
            log_csv_path = Path(log_csv_path)
            log_csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = log_csv_path.exists()

            pt_rel_path = _to_rel_data_path(clip_dir)
            with log_csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["audio_file", "pt_rel_path", "num_words", "num_segments"])
                writer.writerow([audio_path.name, pt_rel_path, num_words, num_segments])

        return saved, num_words

    def process_and_save_from_timestamps_csv_segmentlocal(
        self,
        audio_path: Union[str, Path],
        word_times: Sequence[Sequence[float]],
        out_pt_path: Union[str, Path],
        log_csv_path: Optional[Union[str, Path]] = None,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
        # ==================================================================
        # [ADDED] Optional segments_sec override.
        # If provided, we DO NOT create segments from word_times.
        # This preserves the old caller interface but supports your new flow.
        # ==================================================================
        segments_sec: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Tuple[int, int]:
        """
        Offline convenience wrapper.

        NOTE: Signature/name unchanged to avoid breaking callers.

        [MODIFIED]
        - If segments_sec is provided (from video .pt), we bypass ALL
          word_time-based segmentation and slice audio directly.
        """
        # ==================================================================
        # [MODIFIED] New primary path: use segments_sec if provided
        # ==================================================================
        if segments_sec is not None:
            return self.process_and_save_from_segments_sec_segmentlocal(
                audio_path=audio_path,
                segments_sec=segments_sec,
                out_pt_path=out_pt_path,
                log_csv_path=log_csv_path,
            )

        # ------------------------------------------------------------------
        # Legacy behavior kept intact (only used if segments_sec is NOT provided)
        # ------------------------------------------------------------------
        audio_path = Path(audio_path)
        out_pt_path = Path(out_pt_path)

        num_words = len(word_times)

        mel_segments, segments = self.process_file_with_word_segments_segmentlocal(
            path=str(audio_path),
            word_times=word_times,
            min_factor=min_factor,
            max_factor=max_factor,
        )
        num_segments = len(mel_segments)

        if len(segments) != num_segments:
            raise ValueError(f"Mismatch: mel_segments={num_segments}, segments_sec={len(segments)}")

        for idx, mel in enumerate(mel_segments):
            if not isinstance(mel, torch.Tensor):
                raise TypeError(f"mel_segments[{idx}] is not a Tensor (got {type(mel)})")
            if mel.ndim != 2:
                raise ValueError(f"Expected mel_segments[{idx}] shape (n_mels, T), got {tuple(mel.shape)}")
            if mel.shape[0] != self.n_mels:
                raise ValueError(f"mel_segments[{idx}].shape[0]={mel.shape[0]} expected {self.n_mels}")
            if mel.dtype != torch.float32:
                mel_segments[idx] = mel.float()

        out_pt_path.parent.mkdir(parents=True, exist_ok=True)

        # save_payload = {
        #     "audio_file": audio_path.name,
        #     "mel_segments": mel_segments,
        #     "segments_sec": segments,
        # }
        torch.save(mel_segments, out_pt_path)

        if log_csv_path is not None:
            log_csv_path = Path(log_csv_path)
            log_csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = log_csv_path.exists()

            pt_rel_path = _to_rel_data_path(out_pt_path)
            with log_csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["audio_file", "pt_rel_path", "num_words", "num_segments"])
                writer.writerow([audio_path.name, pt_rel_path, num_words, num_segments])

        return num_segments, num_words
