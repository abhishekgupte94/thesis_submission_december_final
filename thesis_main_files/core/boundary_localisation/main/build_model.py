# ============================================================
# LAV-DF Inference Wrapper (Inference ONLY)
# - Builds model
# - Loads checkpoint
# - Runs forward pass
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
# ---- import from repo ----
# --------------------------------------------------------------------------------------
# Project root helper
# --------------------------------------------------------------------------------------
def _get_project_root(anchor: Optional[Path] = None) -> Path:
    anchor = anchor or Path(__file__).resolve()
    for p in [anchor, *anchor.parents]:
        if p.name == "thesis_main_files":
            return p
    raise RuntimeError("Could not locate project root folder named 'thesis_main_files'.")


# --------------------------------------------------------------------------------------
# Ensure VST repo is importable as a PACKAGE (critical fix)
def _ensure_import_on_syspath() -> None:
    """
    Robust LAV-DF import helper.

    Adds (in priority order):
      1) <repo_root>/external/LAV-DF
      2) <repo_root>/external/LAV-DF/model
      3) <repo_root>/external/LAV-DF/dataset
      4) Fallback: any folder under <repo_root>/external that looks like LAV-DF
    """
    import sys
    from pathlib import Path

    def _add(p: Path) -> None:
        p = p.resolve()
        if p.is_dir():
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)

    # 1) Find thesis_main_files root robustly (cwd can lie under Lightning)
    anchor = Path(__file__).resolve()
    repo_root = None
    for p in [anchor, *anchor.parents]:
        if p.name == "thesis_main_files":
            repo_root = p
            break
    if repo_root is None:
        repo_root = Path.cwd().resolve()

    # 2) Preferred direct location
    direct = repo_root / "external" / "LAV-DF"

    def _add_lavdf_tree(base: Path) -> bool:
        # Heuristic: this looks like the official repo if it has model/ and train.py (or inference.py)
        if not base.is_dir():
            return False
        if not (base / "model").is_dir():
            return False

        # Add repo root first (enables `import model...` if needed)
        _add(base)
        # Add subfolders to enable `from batfd_plus import ...` style imports
        _add(base / "model")
        _add(base / "dataset")
        return True

    if _add_lavdf_tree(direct):
        return

    # 3) Fallback: scan external/ for any folder that looks like LAV-DF
    ext = repo_root / "external"
    if ext.is_dir():
        for cand in sorted(ext.iterdir()):
            if _add_lavdf_tree(cand):
                return

    raise ImportError(
        "Could not locate a valid LAV-DF repo under <repo_root>/external/. "
        "Expected something like external/LAV-DF/{model,dataset,...}"
    )





from pathlib import Path
from typing import Literal, Optional, Dict, Any

import torch
import torch.nn as nn


ModelType = Literal["batfd", "batfd_plus"]


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


class BATFDInferenceWrapper(nn.Module):
    """
    Inference-only wrapper for ControlNet/LAV-DF baseline models:
      - BA-TFD  -> Batfd
      - BA-TFD+ -> BatfdPlus

    Loads either:
      (A) Lightning checkpoint with 'state_dict' + 'hyper_parameters'
      (B) Plain state_dict (.pth) (will instantiate with default args)
    """

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

        # Import from the repo package structure:
        # repo has `model/__init__.py` exporting Batfd, BatfdPlus (see evaluate.py)
        _ensure_import_on_syspath()
w
        # from batfd_plus import BatfdPlus

        import model.BatfdPlus as BatfdPlus
        import model.Batfd as BatfdPlus
        # from model import Batfd, BatfdPlus  # type: ignore

        ModelCls = BatfdPlus if model_type == "batfd_plus" else Batfd

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")

        # ---- Preferred: Lightning restore (uses saved hyperparameters) ----
        if hasattr(ModelCls, "load_from_checkpoint") and isinstance(ckpt, dict) and "state_dict" in ckpt:
            # This recreates the module with the exact hparams saved by PL (best match)
            self.model = ModelCls.load_from_checkpoint(self.checkpoint_path, map_location="cpu")
        else:
            # ---- Fallback: plain state_dict ----
            self.model = ModelCls()  # defaults from __init__
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

            # common prefixes in PL / wrapped checkpoints
            if isinstance(state_dict, dict):
                state_dict = _strip_prefix_if_present(state_dict, "model.")
                state_dict = _strip_prefix_if_present(state_dict, "net.")
                state_dict = _strip_prefix_if_present(state_dict, "module.")

            self.model.load_state_dict(state_dict, strict=strict)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, video: torch.Tensor, audio: torch.Tensor):
        """
        video/audio tensors must match the repo's expected shapes from the DataModule.
        This returns the raw tuple produced by Batfd/BatfdPlus.forward().
        """
        video = video.to(self.device, non_blocking=True)
        audio = audio.to(self.device, non_blocking=True)
        return self.model(video, audio)

    @torch.no_grad()
    def infer_frame_scores(self, video: torch.Tensor, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convenience: returns per-frame classification logits (if present).
        In BatfdPlus.forward() this is: v_frame_cla, a_frame_cla among returned items.
        """
        out = self.forward(video, audio)

        # BatfdPlus returns:
        # (... many maps ... , v_frame_cla, a_frame_cla, v_features, a_features, ...)
        # We grab the two frame classifiers by position:
        # v_frame_cla = out[15], a_frame_cla = out[16] in batfd_plus.py
        # (Same idea holds for batfd.py, but exact tuple length differs slightly.)
        v_frame_cla = out[15]
        a_frame_cla = out[16]
        return {"v_frame_logits": v_frame_cla, "a_frame_logits": a_frame_cla}
