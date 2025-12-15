# pretrain_architecture.py

"""
AV pretraining architecture

This module defines the high-level AVPretrainArchitecture that wraps:
    - A feature extractor backbone (e.g. Swin)
    - Module A: VACL-based head
    - Module A.1: a pre-VACL tokoeinizer stage (will be added later)
    - Module B: common space / EC (CPE) head

NOTE: Sections marked with "KEEP YOUR ORIGINAL IMPLEMENTATION HERE"
are placeholders where you should paste your existing logic unchanged.
Only the parts marked [ADDED] / [MODIFIED] correspond to the changes
we agreed to (config object + FLOPs helper + device-agnostic design).
"""

from __future__ import annotations

from dataclasses import dataclass  # [ADDED]
from typing import Any, Dict, Optional
import torch
from torch import nn, Tensor

# ---------------------------------------------------------------------
# External wrappers (adjust imports to match your repo structure)
# ---------------------------------------------------------------------
# If your actual paths differ, keep using your original imports here.
try:
    from vacl_wrapper import VACLProjectionHead
    from common_projection_head_module_wrapper import FaceAudioCommonSpaceWrapper
except ImportError:
    # You probably already have the correct imports in your real script.
    # Replace this try/except with your original imports.
    VACLProjectionHead = nn.Module
    FaceAudioCommonSpaceWrapper = nn.Module


# ---------------------------------------------------------------------
# [ADDED] Architecture-level configuration
# ---------------------------------------------------------------------
@dataclass
class ArchitectureConfig:  # [ADDED]
    """
    High-level configuration for AVPretrainArchitecture.

    This is intentionally minimal. Extend it as needed.

    Fields
    ------
    audio inputs needed for the Swin2d - (Bxn_melxT)
    video inputs needed for the Swin3d - (Bx3xCxHxW)
    ---> to be loaded from saved .pt files where the video file is saved in a pickle handling and the key within points
        data paths of the vdideo files
       ---> to be loaded from saved .pt files where the audio file is directly saved as the log mel spec in a .pt format
       in a specified audio dir
    vacl_weight:
        Optional weighting for the VACL loss (if you decide to use it
        inside the architecture at some point; currently NOT used here).
    ec_weight:
        Optional weighting for the EC / CPE loss.
    enable_vacl, enable_ec:
        Flags for enabling/disabling submodules (can be handy for
        ablations or debug runs).
    freeze_swin:
        If True, you can implement logic to freeze the Swin backbone
        (e.g. no gradients) externally.
    """
    vacl_weight: float = 1.0
    ec_weight: float = 1.0
    enable_vacl: bool = True
    enable_ec: bool = True
    freeze_swin: bool = False


# ---------------------------------------------------------------------
# High-level architecture
# ---------------------------------------------------------------------
class AVPretrainArchitecture(nn.Module):
    """
    High-level AV pretraining architecture.

    Expects a wrapper that knows how to:
        - Take raw batch dict with audio/video tokens
        - Prepare modality-specific tensors for the feature extractor

    And two heads:
        - `module_a`: VACLProjectionHead (VACL + projection)
        - `module_b`: FaceAudioCommonSpaceWrapper / EC head

    Forward interface (expected):
        batch: Dict[str, Any] with keys like:
            'audio_tokens': (B, T_a, D_a)
            'video_tokens': (B, T_v, D_v)
            plus any meta fields used by the wrapper.

    Returns:
        Dict[str, Any] with keys:
            'audio_features': ...
            'video_features': ...
            'module_a_out': {...}  # should contain "L_vacl" etc.
            'module_b_out': {...}  # should contain "L_cpe" / "L_ec" etc.
    """

    def __init__(
        self,
        av_wrapper: nn.Module,
        swin_backbone: nn.Module,
        module_a: VACLProjectionHead,
        module_b: FaceAudioCommonSpaceWrapper,
        cfg: Optional[ArchitectureConfig] = None,  # [ADDED]
    ) -> None:
        super().__init__()

        # Core submodules (unchanged)
        self.swin_backbone_audio = swin_backbone_2d
        self.swin_backbone_video = swin_backbone_3d
        self.token_unifer = token_unifer
        self.module_a = module_a
        self.module_b = module_b

        # [ADDED] Config object (with defaults)
        self.cfg: ArchitectureConfig = cfg or ArchitectureConfig()

        # ------------------------------------------------------------------
        # KEEP YOUR ORIGINAL __init__ BODY HERE IF YOU HAD EXTRA FIELDS
        # (e.g. additional heads, normalisation layers, pooling ops, etc.)
        #
        # Example:
        #   self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        #   self.some_norm = nn.LayerNorm(d_model)
        #
        # Do not add any hard-coded `.cuda()` / `.to("cuda")` calls here.
        # Device will be managed by Lightning/Trainer.
        # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-level forward:

        1) Run audio/video through respective `swin_backbone` (feature extractor).
        3) Feed the swin backbones outputs into the tokenunifer
        2) Feed the results into module A (VACL head) and module B
           (common-space / EC head).
        3) Return the full dict of features + head outputs.

        NOTE:
        -----
        This method should already exist in your original script.
        This scirpt has to be friendly with the DDP/pytorch lightning
        """

        # ------------------------------------------------------------------
        # KEEP YOUR ORIGINAL FORWARD IMPLEMENTATION HERE
        # ------------------------------------------------------------------

        # Example skeleton (replace with your real body):
        #
        # audio_tokens = batch["audio_tokens"]     # (B, T_a, D_a)
        # video_tokens = batch["video_tokens"]     # (B, T_v, D_v)
        #
        # # Wrapper → feature extractor inputs
        # swin_inputs = self.av_wrapper(
        #     audio_tokens=audio_tokens,
        #     video_tokens=video_tokens,
        #     batch=batch,
        # )
        #
        # # Feature extractor (e.g. Swin)
        # feats = self.swin_backbone(**swin_inputs)
        #
        # # Module A (VACL)
        # module_a_out = self.module_a(
        #     X_v=feats["video_features"],
        #     X_a=feats["audio_features"],
        # )
        #
        # # Module B (Common space / EC head)
        # module_b_out = self.module_b(
        #     X_v=feats["video_features"],
        #     X_a=feats["audio_features"],
        # )
        #
        # out = {
        #     "audio_features": feats["audio_features"],
        #     "video_features": feats["video_features"],
        #     "module_a_out": module_a_out,
        #     "module_b_out": module_b_out,
        # }
        #
        # return out
        # aligner = TemporalTokenAligner(pool_audio_over_freq=True).to(device)
        # aligned = aligner(grid_audio_bchw=grid_aud, grid_viseme_bcdhw=grid_vis, grid_face_bcdhw=grid_face)
        #
        # X_a, X_v, X_f = aligned.X_a, aligned.X_v, aligned.X_f

        raise NotImplementedError(
            "Paste your original AVPretrainArchitecture.forward() body here."
        )

    # ------------------------------------------------------------------
    # [ADDED] FLOPs profiling helper
    # ------------------------------------------------------------------
    def compute_flops(self, batch: Dict[str, Any]) -> int:
        """
        Estimate total FLOPs for a single forward pass on `batch`.

        Usage
        -----
        - Call this from a LightningModule (e.g. AVPretrainSystem),
          ideally on global rank 0 only (trainer.is_global_zero).
        - Do NOT call it inside `training_step` / `validation_step`
          every iteration; it's for one-off profiling.

        This function does *not* affect training behaviour.
        """
        # Local import so fvcore is only required when actually profiling.
        from fvcore.nn import FlopCountAnalysis  # type: ignore[import]

        # FlopCountAnalysis expects (model, *inputs). Your real forward
        # takes a single `batch` dict; we pass it as a single arg tuple.
        fca = FlopCountAnalysis(self, (batch,))
        total_flops = fca.total()
        return int(total_flops)
