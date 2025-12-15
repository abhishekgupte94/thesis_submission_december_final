
import torch
import torch.nn as nn
from typing import Tuple


class LocalFeatureAggregation(nn.Module):
    """
    Local Feature Aggregation (LFA) block for Swin-style backbones.

    Paper-inspired design (NPVForensics, Sec. 3.1, Fig. 3):
      - Two BatchNorms
      - One 3×3 depthwise convolution
      - Two 1×1 pointwise convolutions
      - Operates in a local k×k window (here k=3) over spatial tokens

    Mathematically (simplified from Eqs. (1) and (2) in the paper):

        X_mid = PW_1( DW( BN_1(X_in) ) )
        X_out = PW_2( BN_2(X_mid) )

    where:
        - BN_* are 2D BatchNorm over channels
        - DW is 3×3 depthwise conv (groups = C)
        - PW_* are 1×1 pointwise convs mixing channels

    Expected input:
        x: Tensor of shape (B, L, C), where L = H * W

    This module:
      - reshapes to (B, C, H, W)
      - applies BN/Conv stack
      - reshapes back to (B, L, C)
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        kernel_size: int = 3,
    ) -> None:
        """
        Args:
            dim:              Channel dimension C of the token features.
            input_resolution: (H, W) spatial resolution for the current stage.
            kernel_size:      Local window kernel size for depthwise conv (default: 3).
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.kernel_size = kernel_size

        # BN over channels for (B, C, H, W)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

        # 3×3 depthwise conv: groups = dim → one conv per channel
        self.dw_conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )

        # Two 1×1 pointwise convs to mix channels
        self.pw_conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.pw_conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C) token tensor, with L = H * W.

        Returns:
            out: (B, L, C) tensor after local feature aggregation.
        """
        B, L, C = x.shape
        H, W = self.input_resolution
        if L != H * W:
            raise ValueError(
                f"L={L} does not match H*W={H*W} for input_resolution={self.input_resolution}"
            )
        if C != self.dim:
            raise ValueError(
                f"Channel dim mismatch: got C={C}, expected dim={self.dim}"
            )

        # (B, L, C) -> (B, C, H, W)
        x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Eq. (1): X_mid = PW_1( DW( BN_1(X_in) ) )
        h = self.bn1(x_2d)
        h = self.dw_conv(h)
        h = self.pw_conv1(h)

        # Eq. (2): X_out = PW_2( BN_2(X_mid) )
        h = self.bn2(h)
        h = self.pw_conv2(h)

        # (B, C, H, W) -> (B, L, C)
        out = h.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        return out


