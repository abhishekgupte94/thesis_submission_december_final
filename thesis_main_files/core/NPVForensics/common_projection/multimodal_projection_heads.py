import torch
from torch import nn
from common_space_projector import CommonSpaceProjector


class MultiModalProjectionHeads(nn.Module):
    """
    Implements the Face-Audio projection heads for the Common Space S_fa.

    Paper Reference: Section 3.2 "Evolutionary Consistency Mining"

    Removed Viseme-Audio mappings (g_a->va, g_v->va) as per user constraint.
    """

    def __init__(
            self,
            d_a: int,  # audio feature dim
            d_f: int,  # face feature dim
            d_fa: int,  # common space dim for Face-Audio (S_fa)
    ):
        super().__init__()

        # --- Face -> FA space (g_f->fa) ---
        # The paper specifies a 2-layer projection with ReLU for visual features
        # to handle higher semantic density/granularity.
        self.g_f_to_fa = CommonSpaceProjector(
            in_dim=d_f,
            out_dim=d_fa,
            num_layers=2,  # Inductive bias for visual stream
            use_bn=True,
        )

        # --- Audio -> FA space (g_a->fa) ---
        # The paper specifies a linear projection (1-layer) for audio features.
        self.g_a_to_fa = CommonSpaceProjector(
            in_dim=d_a,
            out_dim=d_fa,
            num_layers=1,  # Inductive bias for audio stream
            use_bn=True,
        )

    def forward(
            self,
            X_f: torch.Tensor,  # (B, T, d_f) or (B, d_f)
            X_a: torch.Tensor,  # (B, T, d_a) or (B, d_a)
    ):
        """
        Projects Face and Audio features into the shared S_fa space.
        """
        Z_f_fa = self.g_f_to_fa(X_f)  # Face   in S_fa
        Z_a_fa = self.g_a_to_fa(X_a)  # Audio  in S_fa

        return {
            "Z_f_fa": Z_f_fa,
            "Z_a_fa": Z_a_fa,
        }