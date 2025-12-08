# ------------------------------------------------------------
# IMPORTS for BoundaryModulePlusTxD
# ------------------------------------------------------------

import torch
from torch import Tensor
from torch import nn

# Import the original BoundaryModulePlus (the BSN++ / PRB module)
# Match your project structure: model/boundary_module_plus.py
from boundary_module_plus import BoundaryModulePlus

class BoundaryModulePlusTxD(BoundaryModulePlus):
    """
    CHANGED WRAPPER around BoundaryModulePlus for inputs shaped (B, T, D_bm_in).

    Original BoundaryModulePlus:
        forward(feature): feature ∈ (B, C_in, T)

    New version:
        forward(feature_bt_d): feature_bt_d ∈ (B, T, D_bm_in)
        Internally: permute → (B, D_bm_in, T), then call super().forward(...)
    """

    def __init__(
        self,
        bm_feature_dim: int,          # D_bm_in = D_enc + 1 (features + frame_logit)
        n_features=(512, 128),
        num_samples: int = 10,
        temporal_dim: int = 512,
        max_duration: int = 40,
    ):
        """
        Args:
            bm_feature_dim: D_bm_in, i.e. encoder_feature_dim + 1
        """
        # CHANGED: pass bm_feature_dim as n_feature_in to the base BoundaryModulePlus
        super().__init__(
            n_feature_in=bm_feature_dim,
            n_features=n_features,
            num_samples=num_samples,
            temporal_dim=temporal_dim,
            max_duration=max_duration,
        )

    def forward(self, feature_bt_d: Tensor):
        """
        Args:
            feature_bt_d: (B, T, D_bm_in) = [encoder features, frame logits]

        Returns:
            confidence_map_p:   (B, D, T)
            confidence_map_c:   (B, D, T)
            confidence_map_p_c: (B, D, T)
        """
        # CHANGED: adapt (B, T, D_bm_in) → (B, D_bm_in, T)
        feature_bct = feature_bt_d.permute(0, 2, 1)  # (B, D_bm_in, T)
        # Call original BoundaryModulePlus.forward, which assumes (B, C_in, T)
        return super().forward(feature_bct)
