# pre_vacl/token_unifier_for_vacl.py
# ============================================================
# DDP / Lightning SAFE unified tokenizer for VACL
#
# Accepts:
#   - video_feat_3d: (B, C_v, T', H', W')
#   - audio_tokens:  (B, L,  C_a)
#
# Produces:
#   - X_v: (B, d_v, S_out)
#   - X_a: (B, d_a, S_out)
#
# Guarantees:
#   ✔ SAME S_out across modalities
#   ✔ DIFFERENT modality dims allowed (d_v != d_a)
#   ✔ NO parameter reassignment in forward()
#   ✔ SAFE for DDP + Lightning
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn


# ============================================================
# Config
# ============================================================

@dataclass
class TokenUnifierForVACLConfig:
    s_out: int = 64
    d_v: int = 256
    d_a: int = 256
    n_heads: int = 4
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # MUST remain False for your current usage
    share_queries: bool = False


# ============================================================
# Learned query unifier (SAFE)
# ============================================================

class LearnedQueryUnifier(nn.Module):
    """
    Unifies variable-length tokens to fixed S_out tokens.

    x: (B, S_in, D_in) -> y: (B, S_out, D_out)

    DDP-safe:
      - Query tensor is a registered Parameter
      - NEVER overwritten during forward
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        s_out: int,
        n_heads: int,
        attn_dropout: float,
        proj_dropout: float,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out),
            nn.Dropout(proj_dropout),
        )

        # Fixed learned queries (SAFE)
        self.q = nn.Parameter(torch.randn(1, s_out, d_out) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S_in, d_in)
        returns: (B, S_out, d_out)
        """
        if x.ndim != 3:
            raise ValueError(
                f"[LearnedQueryUnifier] expected (B,S,D), got {tuple(x.shape)}"
            )

        x = self.in_proj(x)                    # (B, S_in, d_out)
        q = self.q.expand(x.size(0), -1, -1)   # (B, S_out, d_out)

        y, _ = self.attn(q, x, x, need_weights=False)
        y = self.norm(y + q)                   # residual anchor

        return y


# ============================================================
# Pre-VACL unified tokenizer (SAFE)
# ============================================================

class PreVACLTokenUnifier(nn.Module):
    """
    Produces VACL-ready tensors with SAME S_out and DIFFERENT dims.

    Inputs:
      - video_feat_3d: (B, C_v, T', H', W')
      - audio_tokens:  (B, L,  C_a)

    Outputs:
      - X_v: (B, d_v, S_out)
      - X_a: (B, d_a, S_out)
      - aux: debug dict
    """

    def __init__(
        self,
        c_v_in: int,
        c_a_in: int,
        cfg: Optional[TokenUnifierForVACLConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg or TokenUnifierForVACLConfig()

        # Video tokens -> fixed S_out
        self.video_unifier = LearnedQueryUnifier(
            d_in=c_v_in,
            d_out=self.cfg.d_v,
            s_out=self.cfg.s_out,
            n_heads=self.cfg.n_heads,
            attn_dropout=self.cfg.attn_dropout,
            proj_dropout=self.cfg.proj_dropout,
        )

        # Audio tokens -> fixed S_out
        self.audio_unifier = LearnedQueryUnifier(
            d_in=c_a_in,
            d_out=self.cfg.d_a,
            s_out=self.cfg.s_out,
            n_heads=self.cfg.n_heads,
            attn_dropout=self.cfg.attn_dropout,
            proj_dropout=self.cfg.proj_dropout,
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def flatten_video(video_feat_3d: torch.Tensor) -> torch.Tensor:
        """
        (B, C, T', H', W') -> (B, S3D, C)
        """
        if video_feat_3d.ndim != 5:
            raise ValueError(
                f"[PreVACLTokenUnifier] expected (B,C,T,H,W), got {tuple(video_feat_3d.shape)}"
            )

        B, C, Tp, Hp, Wp = video_feat_3d.shape
        return (
            video_feat_3d
            .reshape(B, C, Tp * Hp * Wp)
            .transpose(1, 2)
            .contiguous()
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        video_feat_3d: torch.Tensor,
        audio_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:

        # ---- flatten video tokens
        v_tokens = self.flatten_video(video_feat_3d)   # (B, S3D, C_v)

        if audio_tokens.ndim != 3:
            raise ValueError(
                f"[PreVACLTokenUnifier] expected audio (B,L,C), got {tuple(audio_tokens.shape)}"
            )

        # ---- unify
        Zv = self.video_unifier(v_tokens)    # (B, S_out, d_v)
        Za = self.audio_unifier(audio_tokens) # (B, S_out, d_a)

        # ---- VACL expects channel-first
        X_v = Zv.transpose(1, 2).contiguous()  # (B, d_v, S_out)
        X_a = Za.transpose(1, 2).contiguous()  # (B, d_a, S_out)

        aux = {
            "video_in": tuple(video_feat_3d.shape),
            "audio_in": tuple(audio_tokens.shape),
            "S3D_in": int(v_tokens.shape[1]),
            "Sa_in": int(audio_tokens.shape[1]),
            "S_out": int(self.cfg.s_out),
            "d_v": int(self.cfg.d_v),
            "d_a": int(self.cfg.d_a),
        }

        return X_v, X_a
