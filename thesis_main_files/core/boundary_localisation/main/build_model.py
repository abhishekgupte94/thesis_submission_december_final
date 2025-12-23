# ============================================================
# [PATCH] Embed official Lavdf preprocessing in the wrapper
# - Uses lavdf.py + its utils.read_video/padding/resize
# - Produces exact tensor formats used by the official dataset
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Literal

import torch
import torch.nn as nn

ModelType = Literal["batfd", "batfd_plus"]


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def _ensure_import_on_syspath() -> None:
    import sys
    from pathlib import Path

    def _add(p: Path) -> None:
        p = p.resolve()
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)

    anchor = Path(__file__).resolve()
    repo_root = None
    for p in [anchor, *anchor.parents]:
        if p.name == "thesis_main_files":
            repo_root = p
            break
    if repo_root is None:
        repo_root = Path.cwd().resolve()

    direct = repo_root / "external" / "LAV-DF"

    def _add_lavdf_tree(base: Path) -> bool:
        if not base.is_dir():
            return False
        if not (base / "model").is_dir():
            return False
        _add(base)
        _add(base / "model")
        _add(base / "dataset")
        return True

    if _add_lavdf_tree(direct):
        return

    ext = repo_root / "external"
    if ext.is_dir():
        for cand in sorted(ext.iterdir()):
            if _add_lavdf_tree(cand):
                return

    raise ImportError(
        "Could not locate a valid LAV-DF repo under <repo_root>/external/. "
        "Expected something like external/LAV-DF/{model,dataset,...}"
    )


class BATFDInferenceWrapper(nn.Module):
    """
    Inference-only wrapper for ControlNet/LAV-DF baseline models:
      - BA-TFD  -> Batfd
      - BA-TFD+ -> BatfdPlus

    + [PATCH] Adds official preprocessing from lavdf.py (dataset) so you can feed paths.
    """

    # -----------------------------
    # [PATCH] Official preprocessing config (mirrors Lavdf defaults)
    # -----------------------------
    _DEFAULT_FRAME_PADDING: int = 512
    _DEFAULT_FPS: int = 25
    _DEFAULT_VIDEO_SIZE: Tuple[int, int] = (96, 96)

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_type: ModelType = "batfd_plus",
        device: Optional[str] = None,
        strict: bool = True,
    ):
        super().__init__()

        self.checkpoint_path = str(checkpoint_path)
        self.model_type = model_type
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        _ensure_import_on_syspath()

        # IMPORTANT: your actual repo has lowercase files: model/batfd_plus.py etc
        # Use the lowercase module import (this matches the trace you posted).
        from model.batfd_plus import BatfdPlus  # type: ignore
        from model.batfd import Batfd  # type: ignore

        ModelCls = BatfdPlus if model_type == "batfd_plus" else Batfd

        ckpt: Any = torch.load(self.checkpoint_path, map_location="cpu")

        if hasattr(ModelCls, "load_from_checkpoint") and isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model = ModelCls.load_from_checkpoint(self.checkpoint_path, map_location="cpu")
        else:
            self.model = ModelCls()
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            if not isinstance(state_dict, dict):
                raise TypeError(f"Checkpoint format not understood: {type(state_dict)}")

            state_dict = _strip_prefix_if_present(state_dict, "model.")
            state_dict = _strip_prefix_if_present(state_dict, "net.")
            state_dict = _strip_prefix_if_present(state_dict, "module.")
            self.model.load_state_dict(state_dict, strict=strict)

        self.model.to(self.device)
        self.model.eval()

    # -----------------------------
    # [PATCH] Official LAV-DF preprocessing from lavdf.py
    # -----------------------------
    @staticmethod
    def preprocess_from_paths(
        video_path: str | Path,
        *,
        frame_padding: int = _DEFAULT_FRAME_PADDING,
        fps: int = _DEFAULT_FPS,
        video_size: Tuple[int, int] = _DEFAULT_VIDEO_SIZE,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads + preprocesses a sample EXACTLY like Lavdf.__getitem__:
          - read_video(mp4) -> (video[t,c,h,w], audio[n,1] etc)
          - padding_video to frame_padding
          - padding_audio to frame_padding/fps*16000
          - resize_video to (96,96)
          - rearrange "t c h w -> c t h w"
          - log-mel spec with assert (64,2048)

        Returns:
          video: (1, C, T, H, W)  (batch added)
          audio: (1, 64, 2048)    (batch added)
        """
        # Import *your attached dataset script* (lavdf.py) and its utils
        # Adjust the import path if your file lives elsewhere.
        _ensure_import_on_syspath()
        from lavdf import Lavdf
        # from core.boundary_localisation.main.lavdf import Lavdf  # type: ignore
        from utils import read_video, padding_video, padding_audio, resize_video  # type: ignore
        from einops import rearrange  # type: ignore

        vp = Path(video_path).expanduser().resolve()
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {vp}")

        # read_video returns (video, audio, info/whatever) per your dataset
        video, audio, _ = read_video(str(vp))

        # Apply the SAME padding rules as dataset
        video = padding_video(video, target=frame_padding)
        audio = padding_audio(audio, target=int(frame_padding / fps * 16000))

        # Same resize + layout as dataset: "t c h w -> c t h w"
        video = rearrange(resize_video(video, video_size), "t c h w -> c t h w")

        # Same audio feature as dataset (asserts (64,2048))
        audio = Lavdf._get_log_mel_spectrogram(audio)

        # Add batch dim and return
        return video.unsqueeze(0), audio.unsqueeze(0)

    @torch.no_grad()
    def forward(self, video: torch.Tensor, audio: torch.Tensor):
        video = video.to(self.device, non_blocking=True)
        audio = audio.to(self.device, non_blocking=True)
        return self.model(video, audio)

    # [PATCH] Convenience: infer from file path directly
    @torch.no_grad()
    def infer_from_paths(self, video_path: str | Path) -> Any:
        video, audio = self.preprocess_from_paths(video_path)
        video = video.to(self.device, non_blocking=True)
        audio = audio.to(self.device, non_blocking=True)
        return self.model(video, audio)
