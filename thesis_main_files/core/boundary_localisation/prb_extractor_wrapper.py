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


class BatfdPlusPRBExtractor:
    """
    Phase-1 extractor:
      - loads BatfdPlus Lightning .ckpt
      - returns post-PRB maps (video/audio) + fused post-PRB maps

    Expected inputs:
      video: (B, C, T, H, W)
      audio: (B, 64, 2048)
    Returned maps:
      (B, D, T) for p/c/pc in each of {video, audio, fusion}.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        amp: bool = True,
        strict: bool = True,
    ):
        self.ckpt_path = ckpt_path
        self.amp = amp
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        _ensure_vst_on_syspath()
        from model.batfd_plus import BatfdPlus
        # Lightning load (weights + hparams)
        self.model: BatfdPlus = BatfdPlus.load_from_checkpoint(
            ckpt_path,
            map_location=self.device,
            strict=strict,
        )

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def extract(
        self,
        video: Tensor,
        audio: Tensor,
        return_dict: bool = True,
    ) -> PRBExtraction | Dict[str, Any]:
        """
        Returns post-PRB maps:
          - per modality: video/audio p,c,pc
          - fused: p,c,pc

        NOTE:
        We do NOT call self.model.forward_features() because the official repo’s
        implementation uses torch.column_stack (2D op) which can be brittle if
        tensors are 3D. We reproduce the same intent safely via torch.cat.
        This does not affect checkpoint compatibility (no weights involved).
        """

        video = video.to(self.device, non_blocking=True)
        audio = audio.to(self.device, non_blocking=True)

        use_amp = self.amp and (self.device.startswith("cuda"))
        autocast_dtype = torch.float16  # safe default for inference

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
            # 1) encoders -> (B, C_f, T)
            v_features = self.model.video_encoder(video)
            a_features = self.model.audio_encoder(audio)

            # 2) frame heads -> (B, 1, T)
            v_frame = self.model.video_frame_classifier(v_features)
            a_frame = self.model.audio_frame_classifier(a_features)

            # 3) boundary-module inputs -> (B, C_f+1, T)
            v_bm_in = torch.cat([v_features, v_frame], dim=1)
            a_bm_in = torch.cat([a_features, a_frame], dim=1)

            # 4) POST-PRB per modality (BoundaryModulePlus)
            v_p, v_c, v_pc = self.model.video_boundary_module(v_bm_in)  # each (B, D, T)
            a_p, a_c, a_pc = self.model.audio_boundary_module(a_bm_in)

            # 5) POST-PRB fused boundary maps (ModalFeatureAttnBoundaryMapFusion)
            f_p = self.model.prb_fusion_p(v_bm_in, a_bm_in, v_p, a_p)
            f_c = self.model.prb_fusion_c(v_bm_in, a_bm_in, v_c, a_c)
            f_pc = self.model.prb_fusion_p_c(v_bm_in, a_bm_in, v_pc, a_pc)

        out = PRBExtraction(
            video=PRBMaps(p=v_p, c=v_c, pc=v_pc),
            audio=PRBMaps(p=a_p, c=a_c, pc=a_pc),
            fusion=PRBMaps(p=f_p, c=f_c, pc=f_pc),
        )

        if not return_dict:
            return out

        # dict form is convenient for logging/saving
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