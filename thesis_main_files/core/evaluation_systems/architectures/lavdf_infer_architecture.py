# core/evaluation_systems/architectures/lavdf_infer_architecture.py
# ============================================================
# [NEW | DROP-IN] LAVDFInferArchitecture (ONLINE feature extraction)
#
# NOTE:
#   This file is intentionally a pure nn.Module (no Lightning).
#   It performs official LAV-DF mp4->(video,audio) preprocessing online.
#
# Patch focus (Dec-24):
#   - Guarantee architecture returns "prob_fake" when cfg.return_prob=True,
#     even if model output format differs (tensor/tuple/dict, etc).
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

ModelType = Literal["batfd", "batfd_plus"]


# ============================================================
# [KEPT] Robust LAV-DF import helper (ensure external/LAV-DF is importable)
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
# [KEPT] Official preprocessing (mp4 -> (video,audio))
# ============================================================
def _lavdf_official_preprocess_from_mp4(
    video_mp4_path: Union[str, Path],
    *,
    frame_padding: int = 512,
    fps: int = 25,
    video_size: Tuple[int, int] = (96, 96),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replicates Lavdf.__getitem__ logic:
      video, audio, _ = read_video(mp4)
      video = padding_video(video, target=frame_padding)
      audio = padding_audio(audio, target=int(frame_padding/fps*16000))
      video = rearrange(resize_video(video, (96,96)), "t c h w -> c t h w")
      audio = Lavdf._get_log_mel_spectrogram(audio)  # -> (64,2048)

    Returns:
      video: (C,T,H,W)
      audio: (64,2048)
    """
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
    # Model selection
    model_type: ModelType = "batfd_plus"

    # Online preprocessing params (official defaults)
    frame_padding: int = 512
    fps: int = 25
    video_size: Tuple[int, int] = (96, 96)

    # Output behavior
    return_raw_tuple: bool = False   # include full model output under "raw"
    return_prob: bool = True         # guarantee "prob_fake" when possible


class LAVDFInferArchitecture(nn.Module):
    """
    Single-module architecture for ONLINE inference:
      Preprocess mp4 -> tensors -> LAV-DF model forward -> dict contract
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

        from model.batfd_plus import BatfdPlus  # type: ignore
        from model.batfd import Batfd  # type: ignore

        ModelCls = BatfdPlus if self.cfg.model_type == "batfd_plus" else Batfd
        self.net: nn.Module = ModelCls()

        if ckpt_path is not None:
            ckpt_path = str(ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")

            state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            if not isinstance(state_dict, dict):
                raise TypeError(f"Checkpoint format not understood: {type(state_dict)}")

            for pref in ("model.", "net.", "module."):
                if any(k.startswith(pref) for k in state_dict.keys()):
                    state_dict = {k[len(pref):]: v for k, v in state_dict.items()}

            self.net.load_state_dict(state_dict, strict=bool(strict_load))

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

        return torch.stack(videos, dim=0), torch.stack(audios, dim=0)

    # ============================================================
    # [ADDED] Robust extraction helpers to guarantee "prob_fake"
    # ============================================================
    @staticmethod
    def _try_logits_to_prob_fake(x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Accepts common logits/prob shapes and returns prob_fake of shape (B,).
        Supported:
          - (B,2) -> softmax[:,1]
          - (B,) or (B,1) -> sigmoid
          - (B,T,2) -> mean over T then softmax[:,1]
          - (B,T) or (B,T,1) -> mean over T then sigmoid
        """
        if not torch.is_tensor(x):
            return None

        if x.ndim == 2 and x.shape[-1] == 2:
            return torch.softmax(x, dim=-1)[:, 1]

        if x.ndim == 1:
            return torch.sigmoid(x)

        if x.ndim == 2 and x.shape[-1] == 1:
            return torch.sigmoid(x.squeeze(-1))

        if x.ndim == 3 and x.shape[-1] == 2:
            return torch.softmax(x.mean(dim=1), dim=-1)[:, 1]

        if x.ndim == 3 and x.shape[-1] == 1:
            return torch.sigmoid(x.mean(dim=1).squeeze(-1))

        if x.ndim == 2:  # (B,T) case
            return torch.sigmoid(x.mean(dim=1))

        return None

    @staticmethod
    def _scan_any_for_tensor(obj: Any) -> List[torch.Tensor]:
        """
        Collect tensors from nested (dict/list/tuple) structures.
        """
        found: List[torch.Tensor] = []

        if torch.is_tensor(obj):
            return [obj]

        if isinstance(obj, dict):
            for v in obj.values():
                found.extend(LAVDFInferArchitecture._scan_any_for_tensor(v))
            return found

        if isinstance(obj, (list, tuple)):
            for v in obj:
                found.extend(LAVDFInferArchitecture._scan_any_for_tensor(v))
            return found

        return found

    def forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Normalize mp4 paths
        if "video_paths" in batch:
            mp4_paths = batch["video_paths"]
        elif "video_path" in batch:
            mp4_paths = [batch["video_path"]]
        else:
            raise KeyError("Batch must contain 'video_paths' (list) or 'video_path' (str).")

        if not isinstance(mp4_paths, list) or not all(isinstance(x, str) for x in mp4_paths):
            raise TypeError(f"'video_paths' must be List[str]. Got: {type(mp4_paths)}")

        # Online preprocessing
        video_b, audio_b = self._preprocess_batch_mp4(mp4_paths)

        # Move to module device (since tensors were created inside forward)
        dev = next(self.net.parameters()).device
        video_b = video_b.to(dev, non_blocking=True)
        audio_b = audio_b.to(dev, non_blocking=True)

        # Model forward
        out_raw = self.net(video_b, audio_b)

        out: Dict[str, Any] = {}

        # pass-through metadata if present
        for k in ("clip_ids", "seg_idxs", "video_paths", "audio_paths", "video_rels", "audio_rels", "y"):
            if k in batch:
                out[k] = batch[k]

        # ============================================================
        # [MODIFIED] Guarantee "prob_fake" extraction
        #   - handles tensor / tuple / list / dict outputs
        #   - tries "prob" first if present, else falls back to logits
        # ============================================================
        prob_fake: Optional[torch.Tensor] = None
        logits2: Optional[torch.Tensor] = None

        if self.cfg.return_prob:
            # 1) If dict-like output with obvious prob keys
            if isinstance(out_raw, dict):
                for key in ("prob_fake", "prob", "probs", "p_fake", "fake_prob"):
                    if key in out_raw and torch.is_tensor(out_raw[key]):
                        cand = out_raw[key]
                        # If cand is (B,2) treat as probs; else if (B,) accept
                        if cand.ndim == 2 and cand.shape[-1] == 2:
                            prob_fake = cand[:, 1]
                        elif cand.ndim == 1:
                            prob_fake = cand
                        elif cand.ndim == 2 and cand.shape[-1] == 1:
                            prob_fake = cand.squeeze(-1)
                        if prob_fake is not None:
                            break

            # 2) If not found, scan all tensors and attempt to convert logits->prob
            if prob_fake is None:
                tensors = self._scan_any_for_tensor(out_raw)

                # Prefer (B,2) first
                for t in tensors:
                    if t.ndim == 2 and t.shape[-1] == 2:
                        logits2 = t
                        prob_fake = torch.softmax(t, dim=-1)[:, 1]
                        break

                # Else try other supported shapes
                if prob_fake is None:
                    for t in tensors:
                        pf = self._try_logits_to_prob_fake(t)
                        if pf is not None:
                            prob_fake = pf
                            # If it was (B,2) we'd have caught it above; keep logits2 optional
                            break

            if prob_fake is None:
                # Make the failure actionable (so you can see what the model returned)
                shape_dump = []
                for t in self._scan_any_for_tensor(out_raw):
                    try:
                        shape_dump.append(tuple(t.shape))
                    except Exception:
                        shape_dump.append("<?>")
                raise KeyError(
                    "Could not derive 'prob_fake' from LAV-DF output. "
                    f"out_raw type={type(out_raw)} tensor_shapes={shape_dump}"
                )

            out["prob_fake"] = prob_fake

        # Keep logits2 if we found a clean (B,2)
        if logits2 is not None:
            out["logits2"] = logits2

        # Optional raw
        if self.cfg.return_raw_tuple:
            out["raw"] = out_raw

        return out
