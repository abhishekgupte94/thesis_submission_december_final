# core/training_systems/architectures/lavdf_infer_architecture.py
# ============================================================
# LAV-DF Inference Architecture Wrapper
#
# Purpose:
#   - Load pretrained LAV-DF (batfd / batfd_plus)
#   - Accept audio/video tensors from dataloader
#   - Run forward inference
#   - ALWAYS return:
#       * prob_fake : Tensor[B]
#   - Optionally return:
#       * logits2   : Tensor[B,2]
#
# This file is intentionally ROBUST to different LAV-DF
# output formats (tensor / tuple / dict).
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

# ============================================================
# Config
# ============================================================

ModelType = Literal["batfd", "batfd_plus"]


@dataclass
class LAVDFInferArchitectureConfig:
    model_type: ModelType = "batfd_plus"
    return_prob: bool = True
    return_raw_tuple: bool = False


# ============================================================
# Architecture
# ============================================================

class LAVDFInferArchitecture(nn.Module):
    def __init__(
        self,
        cfg: LAVDFInferArchitectureConfig,
        ckpt_path: str | Path,
        strict_load: bool = False,
    ):
        super().__init__()
        self.cfg = cfg

        # ------------------------------------------------------------
        # Load underlying LAV-DF network
        # ------------------------------------------------------------
        if cfg.model_type == "batfd_plus":
            from core.external.lavdf.models.batfd_plus import BATFDPlus
            self.net = BATFDPlus()
        elif cfg.model_type == "batfd":
            from core.external.lavdf.models.batfd import BATFD
            self.net = BATFD()
        else:
            raise ValueError(f"Unknown model_type={cfg.model_type}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.net.load_state_dict(state, strict=strict_load)

        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

    # ============================================================
    # Robust output parsing helpers
    # ============================================================

    def _as_batch_vec(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Normalize to shape (B,) when possible.
        Accepts (B,) or (B,1).
        """
        if not torch.is_tensor(x):
            return None
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            return x[:, 0]
        return None

    def _prob_from_vec_or_logit(self, v: torch.Tensor) -> torch.Tensor:
        """
        If v looks like probability already -> clamp.
        Else treat as logit -> sigmoid.
        """
        v = v.float()
        if torch.isfinite(v).all():
            mn = float(v.min().detach().cpu())
            mx = float(v.max().detach().cpu())
            if mn >= -1e-3 and mx <= 1.0 + 1e-3:
                return v.clamp(0.0, 1.0)
        return torch.sigmoid(v)

    def _extract_prob_and_logits2(
        self, out_raw: Any
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Tries to extract:
          - logits2: (B,2)
          - prob_fake: (B,)
        from tensor / tuple / list / dict outputs.
        """
        logits2: Optional[torch.Tensor] = None
        prob_fake: Optional[torch.Tensor] = None

        def consider_tensor(t: torch.Tensor) -> None:
            nonlocal logits2, prob_fake

            # (B,2)
            if t.ndim == 2 and t.shape[-1] == 2 and logits2 is None:
                logits2 = t
                prob_fake = torch.softmax(t.float(), dim=-1)[:, 1]
                return

            # (B,T,2)
            if t.ndim == 3 and t.shape[-1] == 2 and logits2 is None:
                logits2 = t.mean(dim=1)
                prob_fake = torch.softmax(logits2.float(), dim=-1)[:, 1]
                return

            # (B,) or (B,1)
            v = self._as_batch_vec(t)
            if v is not None and prob_fake is None:
                prob_fake = self._prob_from_vec_or_logit(v)
                return

        # -------- tensor output
        if torch.is_tensor(out_raw):
            consider_tensor(out_raw)
            return prob_fake, logits2

        # -------- dict output
        if isinstance(out_raw, dict):
            for key in (
                "logits",
                "logits2",
                "pred",
                "prediction",
                "out",
                "y_hat",
                "prob",
                "probs",
                "prob_fake",
            ):
                if key in out_raw and torch.is_tensor(out_raw[key]):
                    consider_tensor(out_raw[key])
                    if prob_fake is not None or logits2 is not None:
                        return prob_fake, logits2

            for v in out_raw.values():
                if torch.is_tensor(v):
                    consider_tensor(v)
                    if prob_fake is not None or logits2 is not None:
                        return prob_fake, logits2

            return prob_fake, logits2

        # -------- tuple / list output
        if isinstance(out_raw, (tuple, list)):
            for x in reversed(out_raw):
                if torch.is_tensor(x) and x.ndim == 2 and x.shape[-1] == 2:
                    consider_tensor(x)
                    return prob_fake, logits2

            for x in reversed(out_raw):
                if torch.is_tensor(x) and x.ndim == 3 and x.shape[-1] == 2:
                    consider_tensor(x)
                    return prob_fake, logits2

            for x in reversed(out_raw):
                if torch.is_tensor(x):
                    v = self._as_batch_vec(x)
                    if v is not None:
                        consider_tensor(x)
                        return prob_fake, logits2

            return prob_fake, logits2

        return prob_fake, logits2

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Expects batch to already contain loaded tensors
        (video, audio) via AVPathsDataModule.
        """

        video_b = batch["video"]
        audio_b = batch["audio"]

        # ------------------------------------------------------------
        # LAV-DF forward
        # ------------------------------------------------------------
        out_raw = self.net(video_b, audio_b)

        out: Dict[str, Any] = {}

        # Pass-through metadata if present
        for k in (
            "clip_ids",
            "seg_idxs",
            "video_paths",
            "audio_paths",
            "video_rels",
            "audio_rels",
            "y",
        ):
            if k in batch:
                out[k] = batch[k]

        # ------------------------------------------------------------
        # Robust probability extraction
        # ------------------------------------------------------------
        prob_fake, logits2 = self._extract_prob_and_logits2(out_raw)

        if logits2 is not None:
            out["logits2"] = logits2

        if self.cfg.return_prob:
            if prob_fake is None:
                raise KeyError(
                    "Could not derive 'prob_fake' from LAV-DF output. "
                    f"out_raw type={type(out_raw)}. "
                    "Inspect tensor shapes in _extract_prob_and_logits2()."
                )
            out["prob_fake"] = prob_fake

        if self.cfg.return_raw_tuple:
            out["raw"] = out_raw

        return out
