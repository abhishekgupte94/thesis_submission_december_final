from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional, Tuple, Dict, Any, Union

import torch
from torch import Tensor


@dataclass
class AVTokenisationConfig:
    """
    High-level configuration wrapper for AV segment tokenisation.
    """
    # Audio
    audio_d_model: int = 256
    audio_max_seq_len: int = 96
    audio_n_mels: int = 80

    # Video
    video_d_model: int = 256
    video_max_seq_len: int = 32
    video_image_size: int = 224
    video_patch_size: int = 4

    # Optional: save tokenised outputs (NOT segments)
    save_dir: Optional[str] = None


class AVSegmentTokenisationWrapper:
    """
    Wrapper that can operate in three modes:

    1) ONLINE (preprocessors):
        - Use audio/video paths + word_times to build segments.
        - Then tokenise.

    2) OFFLINE (.pt segments):
        - Load `mel_stack` + `segment_tensors` from a .pt file.
        - Then tokenise.

    3) DATALOADER / DIRECT TENSORS (training-time):
        - Receive `mel_stack` + `segment_tensors` directly from an external
          DataLoader or caller (no I/O, no preprocessors).
        - Then tokenise.

    In all three cases, the **tokenisation logic is identical**.
    """

    def __init__(
        self,
        av_cfg: Optional[AVTokenisationConfig] = None,
        audio_prep: Optional["AudioPreprocessorNPV"] = None,
        video_prep: Optional["VideoPreprocessorNPV"] = None,
        audio_tokenizer: Optional["AudioModalityTokenizer"] = None,
        video_tokenizer: Optional["VideoModalityTokenizer"] = None,
    ) -> None:
        self.av_cfg = av_cfg or AVTokenisationConfig()

        # ------------------------------------------------------------------
        # Preprocessors (only used in ONLINE mode)
        # ------------------------------------------------------------------
        if audio_prep is None:
            from scripts.preprocessing.audio.AudioPreprocessorNPV import (
                AudioPreprocessorNPV,
                AudioPreprocessorConfig,
            )
            audio_cfg = AudioPreprocessorConfig(
                n_mels=self.av_cfg.audio_n_mels,
                target_num_frames=self.av_cfg.audio_max_seq_len,
            )
            audio_prep = AudioPreprocessorNPV(audio_cfg)

        if video_prep is None:
            from scripts.preprocessing.video.VideoPreprocessorNPV import (
                VideoPreprocessorNPV,
                VideoPreprocessorConfig,
            )
            video_prep = VideoPreprocessorNPV(VideoPreprocessorConfig())

        self.audio_prep = audio_prep
        self.video_prep = video_prep

        # ------------------------------------------------------------------
        # Tokenisers (shared by all modes)
        # ------------------------------------------------------------------
        if audio_tokenizer is None:
            from feature_extraction.not_doing.positional_embedding_stage.audio_positional_embedding import (
                AudioModalityTokenizer,
                AudioModalityConfig,
            )
            a_cfg = AudioModalityConfig(
                d_model=self.av_cfg.audio_d_model,
                max_seq_len=self.av_cfg.audio_max_seq_len,
                n_mels=self.av_cfg.audio_n_mels,
                pad_to_max_seq_len=True,
            )
            audio_tokenizer = AudioModalityTokenizer(a_cfg)

        if video_tokenizer is None:
            from feature_extraction.not_doing.positional_embedding_stage.video_positional_embedding import (
                VideoModalityTokenizer,
                VideoModalityConfig,
            )
            v_cfg = VideoModalityConfig(
                d_model=self.av_cfg.video_d_model,
                max_seq_len=self.av_cfg.video_max_seq_len,
                image_size=self.av_cfg.video_image_size,
                patch_size=self.av_cfg.video_patch_size,
                pad_to_max_seq_len=True,
            )
            video_tokenizer = VideoModalityTokenizer(v_cfg)

        self.audio_tokenizer = audio_tokenizer
        self.video_tokenizer = video_tokenizer

        # Optional save dir (for token outputs, if you want)
        self.save_dir = Path(self.av_cfg.save_dir) if self.av_cfg.save_dir else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # OFFLINE: load precomputed segments from .pt (segments, not tokens)
    # ----------------------------------------------------------------------
    def _load_segments_from_pt(
        self,
        segments_pt_path: Union[str, Path],
    ) -> Tuple[Tensor, List[Tensor], Optional[List[Tuple[float, float]]], Optional[List[Tuple[float, float]]], Dict[str, Any]]:
        """
        Expected minimal structure in .pt file:
            {
                "mel_stack": Tensor(N_a, n_mels, T_a),
                "segment_tensors": List[Tensor(S_i, 3, H, W)],
                # optional:
                "segments_sec_audio": List[(start, end)],
                "segments_sec_video": List[(start, end)],
                "meta": Dict[str, Any],
            }
        """
        segments_pt_path = Path(segments_pt_path)
        data = torch.load(str(segments_pt_path), map_location="cpu")

        if "mel_stack" not in data:
            raise KeyError(f"'mel_stack' not found in {segments_pt_path}")
        if "segment_tensors" not in data:
            raise KeyError(f"'segment_tensors' not found in {segments_pt_path}")

        mel_stack: Tensor = data["mel_stack"]
        segment_tensors: List[Tensor] = data["segment_tensors"]
        segments_sec_audio = data.get("segments_sec_audio", None)
        segments_sec_video = data.get("segments_sec_video", None)

        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            meta = {"meta_raw": meta}

        return mel_stack, segment_tensors, segments_sec_audio, segments_sec_video, meta

    # ----------------------------------------------------------------------
    # NEW: direct tensor path for DataLoader / training-time use
    # ----------------------------------------------------------------------
    def encode_from_tensors(
        self,
        mel_stack: Tensor,
        segment_tensors: List[Tensor],
        segments_sec_audio: Optional[List[Tuple[float, float]]] = None,
        segments_sec_video: Optional[List[Tuple[float, float]]] = None,
        meta: Optional[Dict[str, Any]] = None,
        save_offline: bool = False,
    ) -> Dict[str, Any]:
        """
        Pure tokenizer path: assumes segments are ALREADY preprocessed.

        This is what your "separate dataloader-type function" would call.

        Parameters
        ----------
        mel_stack:
            Tensor (N_segments_a, n_mels, T_a)
        segment_tensors:
            List[Tensor(S_i, 3, H, W)]
        segments_sec_audio, segments_sec_video:
            Optional timing info, if you want to carry it along.
        meta:
            Optional metadata dict (ids, labels, paths, etc.) injected
            by your DataLoader.
        save_offline:
            If True and save_dir is configured, saves token outputs.

        Returns
        -------
        result:
            {
              "audio_tokens": (N_a, T_a_fixed, D_a),
              "video_tokens": (N_v, T_v_fixed, D_v),
              "segments_sec": segments_sec_audio,
              "meta": meta_with_counts_and_optional_saved_path,
            }
        """
        if mel_stack.ndim != 3:
            raise ValueError(
                f"[encode_from_tensors] mel_stack must be (N_segments, n_mels, T), got {mel_stack.shape}"
            )
        if not isinstance(segment_tensors, list) or len(segment_tensors) == 0:
            raise ValueError(
                "[encode_from_tensors] segment_tensors must be a non-empty List[Tensor(S_i, 3, H, W)]."
            )

        # 1) Tokenise audio
        audio_tokens: Tensor = self.audio_tokenizer(mel_batch=mel_stack)

        # 2) Tokenise video
        video_tokens: Tensor = self.video_tokenizer(segment_tensors=segment_tensors)

        # 3) Build meta
        base_meta: Dict[str, Any] = {
            "num_audio_segments": audio_tokens.shape[0],
            "num_video_segments": video_tokens.shape[0],
            "segments_sec_audio": segments_sec_audio,
            "segments_sec_video": segments_sec_video,
        }
        if meta:
            # don't overwrite core keys if they exist in base_meta
            for k, v in meta.items():
                if k not in base_meta:
                    base_meta[k] = v

        result: Dict[str, Any] = {
            "audio_tokens": audio_tokens,
            "video_tokens": video_tokens,
            "segments_sec": segments_sec_audio,
            "meta": base_meta,
        }

        # Optional: save token outputs (if you want this here too)
        if save_offline:
            if self.save_dir is None:
                raise RuntimeError(
                    "[encode_from_tensors] save_offline=True but save_dir is not configured."
                )
            video_id = base_meta.get("video_id", None)
            if video_id is None:
                raise RuntimeError(
                    "[encode_from_tensors] Cannot save token outputs because 'video_id' is missing in meta."
                )

            save_path = self._build_save_path(video_id)
            torch.save(result, save_path)
            result["meta"]["saved_to"] = str(save_path)

        return result

    # ----------------------------------------------------------------------
    # Unified legacy-style API (ONLINE / OFFLINE .pt / DIRECT)
    # ----------------------------------------------------------------------
    def process_sample(
        self,
        # ONLINE mode args:
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        word_times: Optional[Sequence[Sequence[float]]] = None,
        target_clip_duration: Optional[float] = None,
        # common:
        video_id: Optional[str] = None,
        save_offline: bool = False,
        # OFFLINE .pt mode:
        segments_pt_path: Optional[Union[str, Path]] = None,
        # DIRECT tensor mode (if you really want to use process_sample for it):
        mel_stack: Optional[Tensor] = None,
        segment_tensors: Optional[List[Tensor]] = None,
        segments_sec_audio: Optional[List[Tuple[float, float]]] = None,
        segments_sec_video: Optional[List[Tuple[float, float]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        General-purpose entrypoint that supports:

        A) ONLINE mode
            - audio_path, video_path, word_times, target_clip_duration set
            - segments_pt_path is None
            - mel_stack/segment_tensors are None (computed inside).

        B) OFFLINE .pt mode
            - segments_pt_path is set
            - everything else optional / used to override loaded meta.

        C) DIRECT tensor mode
            - mel_stack and segment_tensors provided
            - no paths/word_times needed.
            (In practice, you'll probably call encode_from_tensors() directly.)
        """

        # --------------------------------------------------------------
        # C) DIRECT TENSOR MODE
        # --------------------------------------------------------------
        if mel_stack is not None and segment_tensors is not None:
            # We trust caller to provide correct shapes.
            # We only use audio/video paths / video_id / meta as metadata.
            merged_meta = meta or {}
            if audio_path is not None:
                merged_meta.setdefault("audio_path", audio_path)
            if video_path is not None:
                merged_meta.setdefault("video_path", video_path)
            if video_id is not None:
                merged_meta.setdefault("video_id", video_id)

            return self.encode_from_tensors(
                mel_stack=mel_stack,
                segment_tensors=segment_tensors,
                segments_sec_audio=segments_sec_audio,
                segments_sec_video=segments_sec_video,
                meta=merged_meta,
                save_offline=save_offline,
            )

        # --------------------------------------------------------------
        # B) OFFLINE .pt MODE
        # --------------------------------------------------------------
        if segments_pt_path is not None:
            (
                mel_stack_loaded,
                segment_tensors_loaded,
                seg_sec_a_loaded,
                seg_sec_v_loaded,
                loaded_meta,
            ) = self._load_segments_from_pt(segments_pt_path)

            # If caller passed extra meta, merge it in
            merged_meta = loaded_meta.copy()
            if meta:
                for k, v in meta.items():
                    if k not in merged_meta:
                        merged_meta[k] = v

            # Allow overriding paths/id if provided explicitly
            if audio_path is not None:
                merged_meta["audio_path"] = audio_path
            if video_path is not None:
                merged_meta["video_path"] = video_path
            if video_id is not None:
                merged_meta["video_id"] = video_id

            return self.encode_from_tensors(
                mel_stack=mel_stack_loaded,
                segment_tensors=segment_tensors_loaded,
                segments_sec_audio=seg_sec_a_loaded,
                segments_sec_video=seg_sec_v_loaded,
                meta=merged_meta,
                save_offline=save_offline,
            )

        # --------------------------------------------------------------
        # A) ONLINE MODE (preprocessors)
        # --------------------------------------------------------------
        if audio_path is None or video_path is None:
            raise ValueError(
                "[process_sample] In ONLINE mode, audio_path and video_path must be provided."
            )
        if word_times is None or target_clip_duration is None:
            raise ValueError(
                "[process_sample] In ONLINE mode, word_times and target_clip_duration must be provided."
            )

        # 1) Audio segments
        mel_stack_online, seg_sec_a = self.audio_prep.process_file_with_word_segments(
            path=audio_path,
            word_times=word_times,
            min_factor=0.5,
            max_factor=1.5,
            return_segments=True,
        )
        if mel_stack_online.numel() == 0:
            raise RuntimeError(
                f"[AVSegmentTokenisationWrapper] Audio produced no segments for: {audio_path}"
            )

        # 2) Video segments
        segment_tensors_online, seg_sec_v = self.video_prep.process_video_file_with_word_segments_tensor(
            video_path=video_path,
            cropped_root="",
            word_times=word_times,
            target_clip_duration=target_clip_duration,
            video_id=video_id,
            keep_full_when_no_face=True,
            min_factor=0.5,
            max_factor=1.5,
        )
        if len(segment_tensors_online) == 0:
            raise RuntimeError(
                f"[AVSegmentTokenisationWrapper] Video produced no segments for: {video_path}"
            )

        # 3) Build meta
        online_meta: Dict[str, Any] = meta.copy() if meta else {}
        online_meta.setdefault("audio_path", str(audio_path))
        online_meta.setdefault("video_path", str(video_path))
        if video_id is None:
            video_id = Path(video_path).stem
        online_meta.setdefault("video_id", video_id)

        return self.encode_from_tensors(
            mel_stack=mel_stack_online,
            segment_tensors=segment_tensors_online,
            segments_sec_audio=seg_sec_a,
            segments_sec_video=seg_sec_v,
            meta=online_meta,
            save_offline=save_offline,
        )

    # ----------------------------------------------------------------------
    # Helper: build save path for token outputs
    # ----------------------------------------------------------------------
    def _build_save_path(self, video_id: str) -> Path:
        assert self.save_dir is not None, "save_dir must be set before calling _build_save_path"
        return self.save_dir / f"{video_id}_segments_tokens.pt"
