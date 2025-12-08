# face_audio_common_space_wrapper.py

import torch
from torch import nn

from core.NPVForensics.common_projection.multimodal_projection_heads import MultiModalProjectionHeads
from core.NPVForensics.common_projection.ec_loss import evolutionary_consistency_loss


# face_audio_common_space_wrapper.py

import torch
from torch import nn
import torch.nn.functional as F  # [NEW] for cross-entropy-based InfoNCE

from core.NPVForensics.common_projection.multimodal_projection_heads import MultiModalProjectionHeads


class FaceAudioCommonSpaceWrapper(nn.Module):
    """
    [UPDATED] Wrapper around the Face–Audio common space projection and InfoNCE loss.

    This is designed to be plugged into a larger architecture (e.g. Module B or C)
    and to be Lightning-friendly:

    - Internally uses:
        - MultiModalProjectionHeads (Face/Audio -> S_fa)
        - InfoNCE loss (audio–face homogeneity in common space)
    - Accepts inputs in either:
        - (B, T, D)  -> pooled over T for InfoNCE
        - (B, D)

    Typical usage inside a LightningModule:

        self.module_b = FaceAudioCommonSpaceWrapper(
            d_a=audio_dim,
            d_f=face_dim,
            d_fa=d_b,             # common space dim
            temperature=0.1,
            loss_weight=1.0,
        )

    In training_step:

        out = self.module_b(X_f=face_feats, X_a=audio_feats, compute_ec_loss=True)
        Z_f_fa = out["Z_f_fa"]     # (B, T, d_fa) or (B, d_fa)
        Z_a_fa = out["Z_a_fa"]     # (B, T, d_fa) or (B, d_fa)
        L_info = out["L_info"]     # scalar (if compute_ec_loss=True)
    """

    def __init__(
        self,
        d_a: int,              # audio feature dim
        d_f: int,              # face/visual feature dim
        d_fa: int,             # common FA-space dimension (S_fa)
        temperature: float = 0.1,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()

        # [UNCHANGED] underlying projection heads (Face/Audio -> S_fa)
        self.proj_heads = MultiModalProjectionHeads(
            d_a=d_a,
            d_f=d_f,
            d_fa=d_fa,
        )

        # [UPDATED] InfoNCE hyperparameters
        self.temperature = float(temperature)
        self.loss_weight = float(loss_weight)

        # [NEW] small epsilon for numerical stability
        self.eps = 1e-8

        # [UNCHANGED] Expose output dimensionality (useful for wiring downstream modules)
        self.out_dim = d_fa

    def _pool_for_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        [NEW] Utility for InfoNCE:

        - If z has shape (B, T, D), pool over time -> (B, D)
        - If z has shape (B, D), pass through unchanged.

        This matches the paper's global (identity-level) audio–face homogeneity:
        the loss is computed on clip-level / pooled embeddings, not per segment.
        """
        if z.dim() == 3:
            # Mean over temporal dimension T
            return z.mean(dim=1)  # (B, D)
        elif z.dim() == 2:
            return z
        else:
            raise ValueError(
                f"Expected z of shape (B, D) or (B, T, D), got {z.shape}"
            )

    def _compute_infonce_loss(
        self,
        z_f: torch.Tensor,   # (B, D) face embeddings in common FA space
        z_a: torch.Tensor,   # (B, D) audio embeddings in common FA space
    ) -> torch.Tensor:
        """
        [NEW] Symmetric InfoNCE loss between face and audio embeddings.

        - Positive pairs: (z_f[i], z_a[i]) for each i in batch
        - Negatives: (z_f[i], z_a[j]) for j != i across batch
        - Symmetric: face->audio and audio->face directions averaged

        Returns a scalar tensor.
        """
        assert z_f.shape == z_a.shape, (
            f"z_f and z_a must have same shape, got {z_f.shape} vs {z_a.shape}"
        )

        B, D = z_f.shape
        if B <= 1:
            # Degenerate case: no negatives in batch.
            # Return zero to avoid NaNs and let other losses drive training.
            return z_f.new_zeros(())

        # L2-normalize along feature dimension
        z_f_norm = z_f / (z_f.norm(dim=-1, keepdim=True) + self.eps)  # (B, D)
        z_a_norm = z_a / (z_a.norm(dim=-1, keepdim=True) + self.eps)  # (B, D)

        # Similarity matrices
        # face->audio logits: row i = query face i, columns = all audio
        logits_f2a = torch.matmul(z_f_norm, z_a_norm.t()) / self.temperature  # (B, B)
        # audio->face logits: row i = query audio i, columns = all face
        logits_a2f = torch.matmul(z_a_norm, z_f_norm.t()) / self.temperature  # (B, B)

        # Ground-truth: each sample i matches only sample i in other modality
        labels = torch.arange(B, device=z_f.device)

        # Cross-entropy in both directions
        loss_f2a = F.cross_entropy(logits_f2a, labels)
        loss_a2f = F.cross_entropy(logits_a2f, labels)

        # Symmetric InfoNCE
        loss = 0.5 * (loss_f2a + loss_a2f)
        return loss

    def forward(
        self,
        X_f: torch.Tensor,          # (B, T, d_f) or (B, d_f)
        X_a: torch.Tensor,          # (B, T, d_a) or (B, d_a)
        compute_ec_loss: bool = False,
    ):
        """
        [UPDATED] Forward pass through projection heads + optional InfoNCE loss.

        Parameters
        ----------
        X_f : torch.Tensor
            Face/visual features, shape (B, T, d_f) or (B, d_f).
        X_a : torch.Tensor
            Audio features, shape (B, T, d_a) or (B, d_a).
        compute_ec_loss : bool
            Kept for backwards-compatibility with existing Lightning code.
            If True, computes InfoNCE loss (L_info) between face/audio in S_fa.

        Returns
        -------
        output : dict
            {
                "Z_f_fa":  Face features in common space S_fa (same shape as X_f, last dim = d_fa),
                "Z_a_fa":  Audio features in common space S_fa (same shape as X_a, last dim = d_fa),
                "L_info":  InfoNCE loss (scalar) [if compute_ec_loss=True],
                # "L_ec":  alias to L_info for compatibility
            }
        """

        # 1) Project into common space
        Z = self.proj_heads(X_f=X_f, X_a=X_a)
        Z_f_fa = Z["Z_f_fa"]   # (B, T, d_fa) or (B, d_fa)
        Z_a_fa = Z["Z_a_fa"]   # (B, T, d_fa) or (B, d_fa)

        output = {
            "Z_f_fa": Z_f_fa,
            "Z_a_fa": Z_a_fa,
        }

        # 2) Optional InfoNCE loss (replaces prior EC loss for FA branch)
        if compute_ec_loss:
            # Pool over time if needed -> (B, d_fa)
            z_f_pooled = self._pool_for_loss(Z_f_fa)
            z_a_pooled = self._pool_for_loss(Z_a_fa)

            L_info = self._compute_infonce_loss(
                z_f=z_f_pooled,
                z_a=z_a_pooled,
            )

            # Apply loss weight (so you can tune contribution in main LightningModule)
            L_info = self.loss_weight * L_info

            # Main key
            output["L_info"] = L_info

            # [OPTIONAL] alias, in case your existing code still expects "L_ec"
            output["L_ec"] = L_info

        return output



### USAGE EXAMPLE ------



# out_fa = self.face_audio_wrapper(X_f=face_feats, X_a=audio_feats, compute_ec_loss=True)
# L_info = out_fa["L_info"]
# self.log("train/L_info_fa", L_info)
