# ------------------------------------------------------------
# IMPORTS for FrameLogisticRegressionTxD
# ------------------------------------------------------------

import torch
from torch import Tensor
from torch import nn

# Conv1d utility EXACTLY as used by the original project
from utils import Conv1d

class FrameLogisticRegressionTxD(nn.Module):
    """
    CHANGED VERSION of FrameLogisticRegression for encoders that output (B, T, D_enc).

    Original version:
        Input:  F_m: (B, C_f, T)
        Output: Y^:  (B, 1, T)

    New version:
        Input:  features: (B, T, D_enc)
        Internally: permute → (B, D_enc, T) for Conv1d
        Output: frame_logits: (B, 1, T)  (same as original API for downstream code)
    """

    def __init__(self, n_features: int):
        """
        Args:
            n_features: D_enc (last dimension of encoder output)
        """
        super().__init__()
        # Same Conv1d as before, but we now treat n_features as D_enc.
        self.lr_layer = Conv1d(n_features, 1, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, T, D_enc) from encoder

        Returns:
            frame_logits: (B, 1, T)
        """
        # CHANGED: adapt (B, T, D_enc) → (B, D_enc, T) for Conv1d
        features_ch_first = features.permute(0, 2, 1)          # (B, D_enc, T)
        frame_logits = self.lr_layer(features_ch_first)        # (B, 1, T)
        return frame_logits
