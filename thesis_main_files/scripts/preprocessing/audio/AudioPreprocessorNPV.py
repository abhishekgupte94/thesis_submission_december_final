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
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def build_segments_from_word_times(
    word_times: Sequence[Sequence[float]],
    target_clip_duration: float,
    min_factor: float = 0.5,
    max_factor: float = 1.5,
) -> List[Tuple[float, float]]:
    if not word_times:
        return []

    segments: List[Tuple[float, float]] = [(float(s), float(e)) for s, e in word_times]

    def duration(seg: Tuple[float, float]) -> float:
        return seg[1] - seg[0]

    merged: List[Tuple[float, float]] = []
    current = segments[0]
    for s, e in segments[1:]:
        if duration(current) < min_factor * target_clip_duration:
            current = (current[0], e)
        else:
            merged.append(current)
            current = (s, e)
    merged.append(current)

    final_segments: List[Tuple[float, float]] = []
    for s, e in merged:
        d = e - s
        if d <= max_factor * target_clip_duration:
            final_segments.append((s, e))
        else:
            n_chunks = max(int(round(d / target_clip_duration)), 1)
            chunk_dur = d / n_chunks
            for i in range(n_chunks):
                cs = s + i * chunk_dur
                ce = min(e, cs + chunk_dur)
                if ce > cs:
                    final_segments.append((cs, ce))

    return final_segments


@dataclass
class AudioPreprocessorConfig:
    target_sr: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    target_num_frames: int = 250
    eps: float = 1e-6
    normalize_utterance: bool = True


class AudioPreprocessorNPV:
    def __init__(self, cfg: Optional[AudioPreprocessorConfig] = None) -> None:
        self.cfg = cfg or AudioPreprocessorConfig()

        self.target_sr = self.cfg.target_sr
        self.n_fft = self.cfg.n_fft
        self.hop_length = self.cfg.hop_length
        self.n_mels = self.cfg.n_mels
        self.target_num_frames = self.cfg.target_num_frames
        self.eps = self.cfg.eps
        self.normalize_utterance = self.cfg.normalize_utterance

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=True,
            power=2.0,
        )

    def _load_audio_file(self, path: Union[str, Path]) -> Tuple[Tensor, int]:
        wav, sr = torchaudio.load(str(path))
        return wav, sr

    def _standardize_waveform(self, wav: Tensor, sr: int) -> Tensor:
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.target_sr)

        if self.normalize_utterance:
            peak = wav.abs().max()
            if peak > 0:
                wav = wav / peak

        return wav.squeeze(0)

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
        return_segments: bool = False,
    ):
        wav, sr = self._load_audio_file(path)
        wav = self._standardize_waveform(wav, sr)

        segments_sec = build_segments_from_word_times(
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        if not segments_sec:
            mel_stack = torch.empty(0, self.n_mels, self.target_num_frames)
            return (mel_stack, []) if return_segments else mel_stack

        clips = self.slice_waveform_with_segments(wav, segments_sec)
        mel_list: List[Tensor] = [self._waveform_to_logmel(clip) for clip in clips]

        mel_stack = torch.stack(mel_list, dim=0) if mel_list else torch.empty(0, self.n_mels, self.target_num_frames)
        return (mel_stack, segments_sec) if return_segments else mel_stack

    def process_file_with_word_segments_segmentlocal(
        self,
        path: Union[str, Path],
        word_times: Sequence[Sequence[float]],
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> Tuple[List[Tensor], List[Tuple[float, float]]]:
        if not word_times:
            return [], []

        target_clip_duration = (self.cfg.target_num_frames * self.cfg.hop_length / self.cfg.target_sr)

        mel_stack, segments = self.process_file_with_word_segments(
            path=path,
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            min_factor=min_factor,
            max_factor=max_factor,
            return_segments=True,
        )

        mel_segments: List[Tensor] = [mel_stack[i] for i in range(mel_stack.shape[0])]
        return mel_segments, segments

    def process_and_save_from_timestamps_csv_segmentlocal(
        self,
        audio_path: Union[str, Path],
        word_times: Sequence[Sequence[float]],
        out_pt_path: Union[str, Path],
        log_csv_path: Optional[Union[str, Path]] = None,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
    ) -> Tuple[int, int]:
        """
        Offline convenience wrapper.

        NOTE: Signature/name unchanged to avoid breaking callers.
        """
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

        # ==================================================================
        # [MODIFIED] Save ONLY the 3 requested keys (and nothing else)
        #           - audio_file
        #           - mel_segments
        #           - segments_sec
        # ==================================================================
        save_payload = {
            "audio_file": audio_path.name,
            "mel_segments": mel_segments,
            "segments_sec": segments,
        }
        torch.save(save_payload, out_pt_path)

        # NOTE: logging behavior unchanged; just logs minimal info.
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
