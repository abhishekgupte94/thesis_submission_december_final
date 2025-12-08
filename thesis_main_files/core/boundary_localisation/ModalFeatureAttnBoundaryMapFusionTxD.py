# ------------------------------------------------------------
# IMPORTS for ModalFeatureAttnBoundaryMapFusionTxD
# ------------------------------------------------------------

import torch
from torch import Tensor
from torch import nn

# Base fusion modules
# (Match your project structure: model/fusion_module.py)
from fusion_module import (
    ModalFeatureAttnBoundaryMapFusion,
    ModalFeatureAttnCfgFusion,
)

class ModalFeatureAttnBoundaryMapFusionTxD(ModalFeatureAttnBoundaryMapFusion):
    """
    CHANGED WRAPPER around ModalFeatureAttnBoundaryMapFusion for encoder features (B, T, D_enc).

    Original ModalFeatureAttnBoundaryMapFusion.forward:
        Input:
            video_feature: (B, C_v, T)
            audio_feature: (B, C_a, T)
            video_bm:      (B, D, T)
            audio_bm:      (B, D, T)
        Output:
            fusion_bm:     (B, D, T)

    New version:
        Input:
            video_feature_bt_d: (B, T, D_v)
            audio_feature_bt_d: (B, T, D_a)
            video_bm:           (B, D, T)   # unchanged
            audio_bm:           (B, D, T)   # unchanged

        Internally:
            1) Permute features to channels-first: (B, D_v, T), (B, D_a, T)
            2) Call super().forward(...)
    """

    def forward(
        self,
        video_feature_bt_d: Tensor,
        audio_feature_bt_d: Tensor,
        video_bm: Tensor,
        audio_bm: Tensor,
    ) -> Tensor:
        # CHANGED: adapt (B, T, D) → (B, D, T) for Conv1d inside fusion
        v_feat_bct = video_feature_bt_d.permute(0, 2, 1)   # (B, D_v, T)
        a_feat_bct = audio_feature_bt_d.permute(0, 2, 1)   # (B, D_a, T)

        # Use original fusion logic on channels-first features
        fusion_bm = super().forward(v_feat_bct, a_feat_bct, video_bm, audio_bm)
        return fusion_bm


class ModalFeatureAttnCfgFusionTxD(ModalFeatureAttnCfgFusion):
    """
    CHANGED WRAPPER around ModalFeatureAttnCfgFusion for encoder features (B, T, D_enc).

    Original ModalFeatureAttnCfgFusion.forward:
        Input:
            video_feature: (B, C_v, T)
            audio_feature: (B, C_a, T)
            video_cfg:     (B, T) or (B,) then unsqueezed to (B, 1, T)
            audio_cfg:     (B, T) or (B,) then unsqueezed to (B, 1, T)
        Output:
            fusion_cfg:    (B, T)

    New version:
        Input:
            video_feature_bt_d: (B, T, D_v)
            audio_feature_bt_d: (B, T, D_a)
            video_cfg:          (B, T)
            audio_cfg:          (B, T)

        Internally:
            1) Permute features (B, T, D) → (B, D, T)
            2) Call super().forward(...)
    """
