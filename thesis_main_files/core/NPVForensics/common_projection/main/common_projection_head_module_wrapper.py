# common_projection_head_module_wrapper.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn
import torch.nn.functional as F

from core.NPVForensics.common_projection.multimodal_projection_heads import MultiModalProjectionHeads


# ============================================================
# Pooling helper: accepts (N,D) or (N,S,D) and returns (N,D)
# ============================================================

def _ensure_pooled(x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Ensures x is pooled to (N,D).

    x:
      - (N,D)   -> returned as-is
      - (N,S,D) -> mean pool over S (or masked mean if lengths provided)

    lengths:
      - (N,) number of valid tokens per segment (only used if x is 3D)

    Returns:
      - (N,D)
    """
    if x.dim() == 2:
        return x

    if x.dim() != 3:
        raise ValueError(f"_ensure_pooled expects (N,D) or (N,S,D). Got {tuple(x.shape)}")

    N, S, D = x.shape

    if lengths is None:
        return x.mean(dim=1)

    if lengths.dim() != 1 or lengths.numel() != N:
        raise ValueError(f"lengths must be (N,), got {tuple(lengths.shape)}")

    lengths = lengths.to(device=x.device)
    idx = torch.arange(S, device=x.device).unsqueeze(0).expand(N, S)
    mask = idx < lengths.unsqueeze(1)  # (N,S) bool

    denom = lengths.clamp_min(1).to(x.dtype).unsqueeze(1)  # (N,1)
    pooled = (x * mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom
    return pooled


# ============================================================
# Symmetric InfoNCE for Face <-> Audio
# ============================================================

def face_audio_infonce(Z_a: torch.Tensor, Z_f: torch.Tensor, tau: float) -> Dict[str, torch.Tensor]:
    """
    Symmetric InfoNCE between Audio and Face embeddings.

    Z_a: (N, D_common)
    Z_f: (N, D_common)

    logits_a2f[i,j] = cos(Z_a[i], Z_f[j]) / tau
    positives are diagonal (i,i).
    """
    if Z_a.dim() != 2 or Z_f.dim() != 2:
        raise ValueError(f"InfoNCE expects (N,D). Got {Z_a.shape}, {Z_f.shape}")
    if Z_a.shape != tau and False:  # no-op guard; keeps lint calm if you tweak
        pass
    if Z_a.shape != Z_f.shape:
        raise ValueError(f"Z_a and Z_f must match shape. Got {Z_a.shape} vs {Z_f.shape}")

    # Normalize => dot product equals cosine similarity
    Z_a = F.normalize(Z_a, p=2, dim=1)
    Z_f = F.normalize(Z_f, p=2, dim=1)

    logits_a2f = (Z_a @ Z_f.t()) / tau  # (N,N)
    logits_f2a = (Z_f @ Z_a.t()) / tau  # (N,N)

    targets = torch.arange(Z_a.size(0), device=Z_a.device)

    loss_a = F.cross_entropy(logits_a2f, targets)
    loss_f = F.cross_entropy(logits_f2a, targets)

    return {
        "L_info": loss_a + loss_f,
        "logits_a2f": logits_a2f,
        "logits_f2a": logits_f2a,
    }


# ============================================================
# Wrapper config + module
# ============================================================

@dataclass
class FaceAudioInfoNCEWrapperConfig:
    d_a: int
    d_f: int
    d_common: int = 256
    tau: float = 0.07
    loss_weight: float = 1.0
    force_float32: bool = True


class FaceAudioCommonSpaceWrapper(nn.Module):
    """
    Face-Audio Common Space + InfoNCE.

    IMPORTANT (per your instruction):
      - We are NOT using evolutionary_consistency_loss here.
      - We ONLY compute face-audio InfoNCE.

    Inputs can be either:
      A) token-level per segment:
         audio_in: (N,S,D_a), face_in: (N,S,D_f) with optional lengths
      B) already pooled per segment:
         audio_in: (N,D_a),   face_in: (N,D_f)

    Returns dict:
      - X_a, X_f pooled embeddings (N,D_a)/(N,D_f)
      - Z_a, Z_f projected embeddings (N,D_common)
      - L_info (scalar)
      - logits matrices (N,N)
    """

    def __init__(self, cfg: FaceAudioInfoNCEWrapperConfig):
        super().__init__()
        self.cfg = cfg
        self.tau = float(cfg.tau)

        self.proj_heads = MultiModalProjectionHeads(
            d_a=cfg.d_a,
            d_f=cfg.d_f,
            d_common=cfg.d_common
        )

    def forward(
        self,
        *,
        audio_in: torch.Tensor,              # (N,D_a) or (N,S,D_a)
        face_in: torch.Tensor,               # (N,D_f) or (N,S,D_f)
        audio_lengths: Optional[torch.Tensor] = None,  # (N,) if audio_in is 3D
        face_lengths: Optional[torch.Tensor] = None,   # (N,) if face_in is 3D
        compute_infonce: bool = True,
    ) -> Dict[str, torch.Tensor]:

        # Force float32 (you said no AMP; keeps BN + CE stable)
        if self.cfg.force_float32:
            audio_in = audio_in.float()
            face_in = face_in.float()

        # 1) Pool to (N,D) if token-level was provided
        X_a = _ensure_pooled(audio_in, audio_lengths)  # (N,D_a)
        X_f = _ensure_pooled(face_in, face_lengths)    # (N,D_f)

        if X_a.size(0) != X_f.size(0):
            raise ValueError(f"N mismatch: audio N={X_a.size(0)} vs face N={X_f.size(0)}")

        # 2) Project to common space (N,D_common)
        proj = self.proj_heads(X_a=X_a, X_f=X_f)
        Z_a, Z_f = proj["Z_a"], proj["Z_f"]

        out: Dict[str, torch.Tensor] = {
            "X_a": X_a,
            "X_f": X_f,
            "Z_a": Z_a,
            "Z_f": Z_f,
        }

        # 3) Face-Audio InfoNCE only (NO evolutionary consistency loss)
        if compute_infonce:
            info = face_audio_infonce(Z_a, Z_f, tau=self.tau)
            out["L_info"] = self.cfg.loss_weight * info["L_info"]
            out["logits_a2f"] = info["logits_a2f"]
            out["logits_f2a"] = info["logits_f2a"]

        return out


# ============================================================
# Tiny sanity run (optional)
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = FaceAudioInfoNCEWrapperConfig(d_a=256, d_f=256, d_common=256, tau=0.07)
    m = FaceAudioCommonSpaceWrapper(cfg)

    N, S, D = 32, 12, 256
    audio_tokens = torch.randn(N, S, D)
    face_tokens = torch.randn(N, S, D)

    out = m(audio_in=audio_tokens, face_in=face_tokens)
    out["L_info"].backward()

    print("L_info:", float(out["L_info"]))
    print("Z_a:", out["Z_a"].shape, "Z_f:", out["Z_f"].shape)
