# batfd_full_stack_prb_extractor.py
# ============================================================
# [FULL STACK: BA-TFD+ PRB EXTRACTOR | OPTION B]
#
# Objective:
#   Load official BA-TFD+ (BatfdPlus) from a Lightning .ckpt and extract:
#     1) Per-modality post-PRB maps: video/audio {p,c,pc}
#     2) Fused post-PRB maps: fusion {p,c,pc}
#
# What we rely on (OFFICIAL REPO, UNCHANGED):
#   - BatfdPlus class (LightningModule) with:
#       video_encoder, audio_encoder
#       video_frame_classifier, audio_frame_classifier
#       video_boundary_module (BoundaryModulePlus -> PRB)
#       audio_boundary_module (BoundaryModulePlus -> PRB)
#       prb_fusion_p / prb_fusion_c / prb_fusion_p_c (fusion_module)
#
# What we PATCHED for your thesis integration:
#   - [PATCHED] NO fixed T enforcement (no T=512, no T=75)
#   - [PATCHED] NO audio-width enforcement (no mel-width assertions)
#   - [PATCHED] We DO NOT call BatfdPlus.forward() to avoid:
#       CBG / start-end fusion / post-processing paths
#     Instead, we cut exactly at:
#       post-PRB + post-PRB fusion
#
# Inputs expected (from YOUR pipeline/dataloader):
#   - video: (B, 3, T_v, H, W)  uint8 or float
#   - audio: (B, 64, T_a)       float (e.g., 64x96 or 64x300)
#
# NOTE:
#   Encoders may produce features with their own temporal token length T.
#   We simply propagate whatever T comes out, so boundary maps are (B, D, T).
#
# Outputs:
#   - per-modality PRB maps: (B, D, T)
#   - fused PRB maps:        (B, D, T)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

# ============================================================
# [EXISTING] Official repo import — keep the path identical
# to how the checkpoint was trained/saved.
# ============================================================
from model.batfd_plus import BatfdPlus


# ============================================================
# [EXISTING] Typed containers to make downstream usage clean
# ============================================================
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


class BatfdPlusPRBSubArch(nn.Module):
    """
    ============================================================
    [FULL STACK SUB-ARCH MODULE]
    ============================================================

    Drop-in module to embed inside your system architecture.

    - Loads official BatfdPlus from Lightning .ckpt
    - Runs minimal forward path:
        encoders -> frame heads -> boundary modules (PRB) -> PRB fusion
    - Returns per-modality post-PRB maps + fused maps (Option B)
    - Lightning/DDP friendly:
        * no global caches
        * no mutation
        * deterministic (except autocast)
        * safe to instantiate per process in DDP
    """

    def __init__(
        self,
        ckpt_path: str,
        amp: bool = True,
        strict: bool = True,
        freeze: bool = True,
        map_location: str = "cpu",
    ):
        super().__init__()
        self.amp = bool(amp)

        # ====================================================
        # [EXISTING] Lightning checkpoint load.
        # Loads model weights + hyperparams as trained.
        # ====================================================
        self.batfd: BatfdPlus = BatfdPlus.load_from_checkpoint(
            ckpt_path,
            map_location=map_location,
            strict=strict,
        )
        self.batfd.eval()

        # ====================================================
        # [PATCHED] Freeze by default for "sub-arch feature extraction"
        # (you can set freeze=False if you later want fine-tuning)
        # ====================================================
        if freeze:
            for p in self.batfd.parameters():
                p.requires_grad_(False)

    def forward(self, video: Tensor, audio: Tensor) -> PRBExtraction:
        """
        ====================================================
        Forward pass up to post-PRB + fusion (Option B)
        ====================================================

        Steps:
          1) Encoders:
             v_feat = video_encoder(video)  -> (B, 256, T)
             a_feat = audio_encoder(audio)  -> (B, 256, T)

          2) Frame heads:
             v_frame = video_frame_classifier(v_feat) -> (B, 1, T)
             a_frame = audio_frame_classifier(a_feat) -> (B, 1, T)

          3) Boundary-module inputs:
             v_bm_in = concat(v_feat, v_frame) -> (B, 257, T)
             a_bm_in = concat(a_feat, a_frame) -> (B, 257, T)

          4) BoundaryModulePlus (PRB) per modality:
             video_boundary_module(v_bm_in) -> (v_p, v_c, v_pc) each (B, D, T)
             audio_boundary_module(a_bm_in) -> (a_p, a_c, a_pc) each (B, D, T)

          5) PRB fusion (boundary-map space):
             prb_fusion_* returns fused map (B, D, T)
        """

        # ====================================================
        # [PATCHED] No .forward() call on BatfdPlus.
        # We do not want CBG/start-end/post-process.
        # ====================================================

        use_amp = self.amp and video.is_cuda
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # ---------- 1) Encoders ----------
            v_feat = self.batfd.video_encoder(video)
            a_feat = self.batfd.audio_encoder(audio)

            # ---------- 2) Frame heads ----------
            v_frame = self.batfd.video_frame_classifier(v_feat)
            a_frame = self.batfd.audio_frame_classifier(a_feat)

            # ---------- 3) Boundary inputs ----------
            v_bm_in = torch.cat([v_feat, v_frame], dim=1)
            a_bm_in = torch.cat([a_feat, a_frame], dim=1)

            # ---------- 4) PRB per modality ----------
            v_p, v_c, v_pc = self.batfd.video_boundary_module(v_bm_in)
            a_p, a_c, a_pc = self.batfd.audio_boundary_module(a_bm_in)

            # ---------- 5) PRB fusion (Option B) ----------
            f_p = self.batfd.prb_fusion_p(v_bm_in, a_bm_in, v_p, a_p)
            f_c = self.batfd.prb_fusion_c(v_bm_in, a_bm_in, v_c, a_c)
            f_pc = self.batfd.prb_fusion_p_c(v_bm_in, a_bm_in, v_pc, a_pc)

        return PRBExtraction(
            video=PRBMaps(p=v_p, c=v_c, pc=v_pc),
            audio=PRBMaps(p=a_p, c=a_c, pc=a_pc),
            fusion=PRBMaps(p=f_p, c=f_c, pc=f_pc),
        )

    # ============================================================
    # [ADDED] Convenience: dict output for logging/saving
    # ============================================================
    @staticmethod
    def to_dict(out: PRBExtraction) -> Dict[str, Any]:
        return {
            "video": {"p": out.video.p, "c": out.video.c, "pc": out.video.pc},
            "audio": {"p": out.audio.p, "c": out.audio.c, "pc": out.audio.pc},
            "fusion": {"p": out.fusion.p, "c": out.fusion.c, "pc": out.fusion.pc},
        }


# ============================================================
# [OPTIONAL] Lightning wrapper for DDP predict/extraction
# Use this if you want to run extraction as a standalone job.
# ============================================================
try:
    import pytorch_lightning as pl
except Exception:
    pl = None


# if pl is not None:
#     class LightningPRBExtractor(pl.LightningModule):
#         """
#         ============================================================
#         [LIGHTNING/DDP EXTRACTOR MODULE]
#         ============================================================
#         - Use with Trainer.predict() on 8×A100
#         - No optimizer, no training_step
#         - Returns PRB maps per batch
#         """
#
#         def __init__(
#             self,
#             ckpt_path: str,
#             amp: bool = True,
#             strict: bool = True,
#             freeze: bool = True,
#             return_cpu: bool = True,
#         ):
#             super().__init__()
#             self.save_hyperparameters()
#             self.subarch = BatfdPlusPRBSubArch(
#                 ckpt_path=ckpt_path,
#                 amp=amp,
#                 strict=strict,
#                 freeze=freeze,
#                 map_location="cpu",
#             )
#             self.return_cpu = bool(return_cpu)
#
#         def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Tensor]:
#             # Supports dict batches or tuple/list batches
#             if isinstance(batch, dict):
#                 video = batch["video"]
#                 audio = batch["audio"]
#             else:
#                 video = batch[0]
#                 audio = batch[1]
#
#             out = self.subarch(video, audio)
#             out_dict = self.subarch.to_dict(out)
#
#             if self.return_cpu:
#                 for mod in out_dict:
#                     for k in out_dict[mod]:
#                         out_dict[mod][k] = out_dict[mod][k].detach().cpu()
#
#             return out_dict
#
#         def configure_optimizers(self):
#             return None