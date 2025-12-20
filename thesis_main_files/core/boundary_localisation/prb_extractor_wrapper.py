# model/prb_extractor_wrapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch import Tensor
import sys
from pathlib import Path
# IMPORTANT: keep this import path identical to the repo
# from model.batfd_plus import BatfdPlus

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
# --------------------------------------------------------------------------------------
def _ensure_vst_on_syspath() -> None:
    """
    Ensures Video-Swin-Transformer repo root is on sys.path so that
    `mmaction.models.backbones.swin_transformer` is imported in package context.
    """
    project_root = _get_project_root()
    vst_root = project_root / "external" / "LAV-DF"

    vst_root_str = str(vst_root)
    if vst_root_str not in sys.path:
        sys.path.insert(0, vst_root_str)



@dataclass
class PRBMaps:
    # Each tensor is (B, D, T)
    p: Tensor
    c: Tensor
    pc: Tensor


@dataclass
class PRBExtraction:
    video: PRBMaps
    audio: PRBMaps
    fusion: PRBMaps


# integrations/batfdplus_prb_subarch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

# Official repo import (keep path consistent with training)
from model.batfd_plus import BatfdPlus


@dataclass
class PRBMaps:
    # each: (B, D, T)
    p: Tensor
    c: Tensor
    pc: Tensor


@dataclass
class PRBExtraction:
    video: PRBMaps
    audio: PRBMaps
    fusion: PRBMaps


class BatfdPlusPRBSubArch(nn.Module):
    """
    Drop-in sub-architecture for your thesis model.

    - Loads official BatfdPlus from a Lightning .ckpt
    - Runs only up to post-PRB (per-modality) + post-PRB fusion (Option B)
    - DDP-friendly: no global caches, no mutable shared state, pure forward.

    Inputs:
      video: (B, 3, 512, 96, 96)
      audio: (B, 64, 2048)
    Outputs:
      PRBExtraction with maps (B, D, 512)
    """

    def __init__(
        self,
        ckpt_path: str,
        device_map_location: Optional[str] = None,
        amp: bool = True,
        strict: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.amp = amp
        self.strict = strict

        map_location = device_map_location or "cpu"

        # IMPORTANT: instantiate inside each process (DDP will spawn processes)
        # Loading on CPU avoids GPU spike during init; move later in .to(device).

        _ensure_vst_on_syspath()
        from model.batfd_plus import BATfdPlus
        self.batfd: BatfdPlus = BatfdPlus.load_from_checkpoint(
            ckpt_path,
            map_location=map_location,
            strict=strict,
        )

        self.batfd.eval()

        if freeze:
            for p in self.batfd.parameters():
                p.requires_grad_(False)

    def forward(self, video: Tensor, audio: Tensor) -> PRBExtraction:
        # NOTE: do not call batfd.forward() to avoid CBG/start-end/post-processing branches.
        # We cut exactly at PRB + fusion.

        # DDP-safe autocast: use if on CUDA
        use_amp = self.amp and video.is_cuda
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # 1) encoders -> (B, 256, 512) intended
            v_features = self.batfd.video_encoder(video)
            a_features = self.batfd.audio_encoder(audio)

            # 2) frame heads -> (B, 1, 512)
            v_frame = self.batfd.video_frame_classifier(v_features)
            a_frame = self.batfd.audio_frame_classifier(a_features)

            # 3) boundary-module inputs -> (B, 257, 512)
            v_bm_in = torch.cat([v_features, v_frame], dim=1)
            a_bm_in = torch.cat([a_features, a_frame], dim=1)

            # 4) post-PRB per-modality maps -> each (B, D, 512)
            v_p, v_c, v_pc = self.batfd.video_boundary_module(v_bm_in)
            a_p, a_c, a_pc = self.batfd.audio_boundary_module(a_bm_in)

            # 5) post-PRB fused maps -> each (B, D, 512)
            f_p = self.batfd.prb_fusion_p(v_bm_in, a_bm_in, v_p, a_p)
            f_c = self.batfd.prb_fusion_c(v_bm_in, a_bm_in, v_c, a_c)
            f_pc = self.batfd.prb_fusion_p_c(v_bm_in, a_bm_in, v_pc, a_pc)

        return PRBExtraction(
            video=PRBMaps(p=v_p, c=v_c, pc=v_pc),
            audio=PRBMaps(p=a_p, c=a_c, pc=a_pc),
            fusion=PRBMaps(p=f_p, c=f_c, pc=f_pc),
        )

    def as_dict(self, out: PRBExtraction) -> Dict[str, Any]:
        return {
            "video": {"p": out.video.p, "c": out.video.c, "pc": out.video.pc},
            "audio": {"p": out.audio.p, "c": out.audio.c, "pc": out.audio.pc},
            "fusion": {"p": out.fusion.p, "c": out.fusion.c, "pc": out.fusion.pc},
        }



##### USAGE EXAMPLE SCRIPT


# extract_prb_maps.py
# from __future__ import annotations
#
# import argparse
# import torch
#
# from model.prb_extractor_wrapper import BatfdPlusPRBExtractor
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", required=True, type=str, help="Path to BatfdPlus .ckpt")
#     ap.add_argument("--device", default=None, type=str, help="cuda / cpu (optional)")
#     ap.add_argument("--amp", action="store_true", help="Enable autocast on CUDA")
#     args = ap.parse_args()
#
#     extractor = BatfdPlusPRBExtractor(
#         ckpt_path=args.ckpt,
#         device=args.device,
#         amp=args.amp,
#         strict=True,
#     )
#
#     # Dummy inputs with expected shapes:
#     # video: (B, C, T, H, W)
#     # audio: (B, 64, 2048)
#     B, C, T, H, W = 1, 3, 512, 96, 96
#     video = torch.randn(B, C, T, H, W)
#     audio = torch.randn(B, 64, 2048)
#
#     maps = extractor.extract(video=video, audio=audio, return_dict=True)
#
#     # Print shapes
#     for mod in ("video", "audio", "fusion"):
#         for k in ("p", "c", "pc"):
#             t = maps[mod][k]
#             print(f"{mod}.{k}: {tuple(t.shape)}  dtype={t.dtype}  device={t.device}")
#
#
# if __name__ == "__main__":
#     main()
#########