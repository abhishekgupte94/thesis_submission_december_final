# core/training_systems/architectures/lavdf_infer_architecture.py
# ============================================================
# [NEW | DROP-IN] LAVDFInferArchitecture (ONLINE feature extraction)
#
# Why this file exists:
#   - You are NOT doing offline training/export of features.
#   - So the "architecture" must include the official LAV-DF feature extractor
#     (video+audio preprocessing) inside the forward flow.
#   - Mirrors the essential style of your finetune architecture:
#       * pure nn.Module (NO Lightning)
#       * no logging
#       * DDP-safe (rank-independent)
#       * returns a dict contract that a Lightning system can consume
#
# Key simplification (per you):
#   - ONLY ONE module: the LAV-DF model (Batfd/BatfdPlus)
#   - Feature extractor is integrated (official repo logic)
#
# Expected batch input (from your existing AV path dataloader):
#   - Either:
#       batch["video_paths"] : List[str]  (absolute paths to .mp4)
#     and optionally batch["y"] : Tensor[B] (0/1)
#   - OR:
#       batch["video_path"]  : str  (single item)
#
# NOTE:
#   - Official LAV-DF preprocessing reads audio directly from the mp4 via read_video().
#   - So audio_rel/audio_path is kept for bookkeeping but not required here.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

ModelType = Literal["batfd", "batfd_plus"]


# ============================================================
# [KEPT] Robust LAV-DF import helper (same spirit as your wrapper)
# ============================================================
def _ensure_lavdf_import_on_syspath() -> None:
    """
    Adds:
      <repo_root>/external/LAV-DF
      <repo_root>/external/LAV-DF/model
      <repo_root>/external/LAV-DF/dataset
    """
    import sys

    anchor = Path(__file__).resolve()
    repo_root = None
    for p in [anchor, *anchor.parents]:
        if p.name == "thesis_main_files":
            repo_root = p
            break
    if repo_root is None:
        repo_root = Path.cwd().resolve()

    base = repo_root / "external" / "LAV-DF"

    def _add(pp: Path) -> None:
        if pp.is_dir():
            s = str(pp.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)

    if not (base.is_dir() and (base / "model").is_dir()):
        raise ImportError(
            "Could not locate LAV-DF under thesis_main_files/external/LAV-DF. "
            "Expected external/LAV-DF/model to exist."
        )

    _add(base)
    _add(base / "model")
    _add(base / "dataset")


# ============================================================
# [NEW] Official preprocessing (the SAME logic you validated)
# ============================================================
def _lavdf_official_preprocess_from_mp4(
    video_mp4_path: Union[str, Path],
    *,
    frame_padding: int = 512,
    fps: int = 25,
    video_size: Tuple[int, int] = (96, 96),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replicates Lavdf.__getitem__ from your attached lavdf.py:

      video, audio, _ = read_video(mp4)
      video = padding_video(video, target=frame_padding)
      audio = padding_audio(audio, target=int(frame_padding/fps*16000))
      video = rearrange(resize_video(video, (96,96)), "t c h w -> c t h w")
      audio = Lavdf._get_log_mel_spectrogram(audio)  # -> (64,2048)

    Returns:
      video: (C,T,H,W)
      audio: (64,2048)
    """
    # [IMPORTANT] Use YOUR attached dataset script + official utils
    _ensure_lavdf_import_on_syspath()
    from lavdf import Lavdf  # type: ignore
    from utils import read_video, padding_video, padding_audio, resize_video  # type: ignore
    from einops import rearrange  # type: ignore

    vp = Path(video_mp4_path).expanduser().resolve()
    if not vp.exists():
        raise FileNotFoundError(f"Missing mp4: {vp}")

    v, a, _ = read_video(str(vp))
    v = padding_video(v, target=int(frame_padding))
    a = padding_audio(a, target=int(int(frame_padding) / int(fps) * 16000))
    v = rearrange(resize_video(v, video_size), "t c h w -> c t h w")
    a = Lavdf._get_log_mel_spectrogram(a)
    return v, a


# ============================================================
# Config
# ============================================================
@dataclass
class LAVDFInferArchitectureConfig:
    # ------------------------------------------------------------
    # [KEPT] Model selection
    # ------------------------------------------------------------
    model_type: ModelType = "batfd_plus"

    # ------------------------------------------------------------
    # [ADDED] Online preprocessing params (official defaults)
    # ------------------------------------------------------------
    frame_padding: int = 512
    fps: int = 25
    video_size: Tuple[int, int] = (96, 96)

    # ------------------------------------------------------------
    # [ADDED] Output behavior
    # ------------------------------------------------------------
    return_raw_tuple: bool = False   # if True, include full model output under "raw"
    return_prob: bool = True         # include "prob_fake" derived from (B,2) logits when possible


class LAVDFInferArchitecture(nn.Module):
    """
    ============================================================
    [NEW] Single-module architecture for ONLINE inference:
      FeatureExtractor (official) -> LAV-DF (Batfd/BatfdPlus)
    ============================================================

    Notes:
      - Pure nn.Module (NO Lightning)
      - No logging
      - Device handling:
          * Lightning will move incoming batch tensors to device,
            BUT here we build tensors inside forward from file paths.
          * So we must move them onto the module device.
        This is the minimal necessary exception for online feature extraction.
    """

    def __init__(
        self,
        *,
        cfg: Optional[LAVDFInferArchitectureConfig] = None,
        ckpt_path: Optional[Union[str, Path]] = None,
        strict_load: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = LAVDFInferArchitectureConfig() if cfg is None else cfg

        _ensure_lavdf_import_on_syspath()

        # ------------------------------------------------------------
        # [KEPT] Instantiate model class (official filenames are lowercase)
        # ------------------------------------------------------------
        from model.batfd_plus import BatfdPlus  # type: ignore
        from model.batfd import Batfd  # type: ignore

        ModelCls = BatfdPlus if self.cfg.model_type == "batfd_plus" else Batfd
        self.net: nn.Module = ModelCls()

        # ------------------------------------------------------------
        # [ADDED] Optional checkpoint loading
        #   - Keep it minimal: accept either plain state_dict or {"state_dict": ...}
        # ------------------------------------------------------------
        if ckpt_path is not None:
            ckpt_path = str(ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")

            state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            if not isinstance(state_dict, dict):
                raise TypeError(f"Checkpoint format not understood: {type(state_dict)}")

            # common wrappers; harmless even if absent
            for pref in ("model.", "net.", "module."):
                if any(k.startswith(pref) for k in state_dict.keys()):
                    state_dict = {k[len(pref):]: v for k, v in state_dict.items()}

            self.net.load_state_dict(state_dict, strict=bool(strict_load))

    # ============================================================
    # [NEW] Preprocess a batch of mp4 paths -> batched tensors
    # ============================================================
    def _preprocess_batch_mp4(self, mp4_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        videos: List[torch.Tensor] = []
        audios: List[torch.Tensor] = []

        for p in mp4_paths:
            v, a = _lavdf_official_preprocess_from_mp4(
                p,
                frame_padding=self.cfg.frame_padding,
                fps=self.cfg.fps,
                video_size=self.cfg.video_size,
            )
            videos.append(v)  # (C,T,H,W)
            audios.append(a)  # (64,2048)

        # stack: (B,C,T,H,W), (B,64,2048)
        video_b = torch.stack(videos, dim=0)
        audio_b = torch.stack(audios, dim=0)
        return video_b, audio_b

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Expected batch keys (from your AV-path dataloader):
          - "video_paths": List[str]  OR "video_path": str
          - optional: "y" : Tensor[B] (labels)
          - optional: "clip_ids", "seg_idxs" for bookkeeping

        Returns:
          dict with essentials:
            - "logits2" (if we can reliably extract (B,2))
            - "prob_fake" (if enabled and logits2 extracted)
            - "raw" (optional: full model output tuple)
            - "y" passed through if present
        """
        # ------------------------------------------------------------
        # [KEPT] Normalize mp4 path list
        # ------------------------------------------------------------
        if "video_paths" in batch:
            mp4_paths = batch["video_paths"]
        elif "video_path" in batch:
            mp4_paths = [batch["video_path"]]
        else:
            raise KeyError("Batch must contain 'video_paths' (list) or 'video_path' (str).")

        if not isinstance(mp4_paths, list) or not all(isinstance(x, str) for x in mp4_paths):
            raise TypeError(f"'video_paths' must be List[str]. Got: {type(mp4_paths)}")

        # ------------------------------------------------------------
        # [ADDED] Online preprocessing (official)
        # ------------------------------------------------------------
        video_b, audio_b = self._preprocess_batch_mp4(mp4_paths)

        # ------------------------------------------------------------
        # [ADDED] Move to module device (necessary for online-built tensors)
        # ------------------------------------------------------------
        dev = next(self.net.parameters()).device
        video_b = video_b.to(dev, non_blocking=True)
        audio_b = audio_b.to(dev, non_blocking=True)

        # ------------------------------------------------------------
        # [KEPT] Model forward
        # ------------------------------------------------------------
        out_raw = self.net(video_b, audio_b)

        out: Dict[str, Any] = {}

        # pass-through metadata if present
        for k in ("clip_ids", "seg_idxs", "video_paths", "audio_paths", "video_rels", "audio_rels", "y"):
            if k in batch:
                out[k] = batch[k]

        # ------------------------------------------------------------
        # [ADDED] Extract a usable (B,2) logits tensor, if possible
        # (mirrors the robust evaluator idea)
        # ------------------------------------------------------------
        logits2 = None
        if torch.is_tensor(out_raw):
            if out_raw.ndim == 2 and out_raw.shape[-1] == 2:
                logits2 = out_raw
        elif isinstance(out_raw, (tuple, list)):
            # search from end: (B,2) first, else (B,T,2)->mean
            for x in reversed(out_raw):
                if torch.is_tensor(x) and x.ndim == 2 and x.shape[-1] == 2:
                    logits2 = x
                    break
            if logits2 is None:
                for x in reversed(out_raw):
                    if torch.is_tensor(x) and x.ndim == 3 and x.shape[-1] == 2:
                        logits2 = x.mean(dim=1)
                        break

        if logits2 is not None:
            out["logits2"] = logits2
            if self.cfg.return_prob:
                out["prob_fake"] = torch.softmax(logits2, dim=-1)[:, 1]

        # ------------------------------------------------------------
        # [ADDED] Optionally include raw tuple for boundary localization usage
        # ------------------------------------------------------------
        if self.cfg.return_raw_tuple:
            out["raw"] = out_raw

        return out
