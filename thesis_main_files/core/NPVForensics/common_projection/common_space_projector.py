import torch
from torch import nn


class CommonSpaceProjector(nn.Module):
    """
    Generic projection head for mapping modality features to a common space.

    Paper Logic Implementation:
    - Audio (num_layers=1): Linear -> BatchNorm
    - Visual (num_layers=2): Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm

    Ref: Section 3.2, paragraph regarding "Inductive biases in projection".
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            num_layers: int = 1,
            use_bn: bool = True,
            activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        assert num_layers in (1, 2), "NPVForensics only uses 1 or 2 layer projections."

        layers = []
        if num_layers == 1:
            # Linear Projection (Used for Audio streams)
            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
        else:
            # MLP Projection (Used for Visual streams: Face, Viseme)
            # Layer 1
            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(activation())

            # Layer 2
            layers.append(nn.Linear(out_dim, out_dim, bias=True))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))

        self.proj = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Handles (B, D) or (B, T, D) inputs.
        Reshapes B*T into a single batch dimension for BatchNorm1d stability.
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
            x_proj = self.proj(x_flat)  # (B*T, out_dim)
            x_proj = x_proj.reshape(B, T, self.out_dim)
            return x_proj
        elif x.dim() == 2:
            return self.proj(x)
        else:
            raise ValueError(f"Expected input of shape (B, D) or (B, T, D), got {x.shape}")