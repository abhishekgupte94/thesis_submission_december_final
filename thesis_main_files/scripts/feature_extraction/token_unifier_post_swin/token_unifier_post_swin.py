# pre_vacl/token_unifier_for_vacl.py
# Drop-in adapter BEFORE VACL.
#
# Goal:
#   - Accepts:
#       video_feat_3d: (B, C_v, T', H', W')  from Swin3D forward_features()
#       audio_tokens:  (B, L,  C_a)          from Swin2D forward_features() (post-norm tokens)
#   - Produces:
#       X_v: (B, d_v, S_out)  for VACL
#       X_a: (B, d_a, S_out)  for VACL
#
# Key design:
#   - SAME S_out across modalities (required by your VACL math)
#   - SHARED latent query slots across modalities (so token index s means the same latent "slot")
#   - DIFFERENT modality dimensions preserved: d_v != d_a is allowed and supported
#   - NO FFN (just LN -> Linear -> MHA -> residual+LN)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn


@dataclass
class TokenUnifierForVACLConfig:
    s_out: int = 64

    # keep modality dims different (do NOT force equal)
    d_v: int = 256   # output dim for video tokens into VACL
    d_a: int = 256   # output dim for audio tokens into VACL

    n_heads: int = 4
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # Important for your VACL concat J_va[..., s] meaning:
    # use the SAME latent slot basis across modalities.
    share_queries: bool = True


class LearnedQueryUnifier(nn.Module):
    """
    Unify variable-length token sequences to fixed S_out tokens.

    x: (B, S_in, D_in) -> y: (B, S_out, D_out)

    Uses learned query slots (shared across modalities if provided) to ensure
    token index s has consistent "slot identity".
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        s_out: int,
        n_heads: int,
        attn_dropout: float,
        proj_dropout: float,
        shared_queries: Optional[nn.Parameter] = None,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out),
            nn.Dropout(proj_dropout),
        )

        # queries live in the *output* space (d_out)
        if shared_queries is None:
            self.q = nn.Parameter(torch.randn(1, s_out, d_out) * 0.02)
        else:
            # MUST match d_out
            if shared_queries.shape[-1] != d_out:
                raise ValueError(
                    f"[LearnedQueryUnifier] shared_queries has last-dim {shared_queries.shape[-1]} "
                    f"but d_out={d_out}. Provide separate shared_q per modality-dim."
                )
            self.q = shared_queries

        self.attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"[LearnedQueryUnifier] expected (B,S,D), got {tuple(x.shape)}")

        x = self.in_proj(x)                  # (B, S_in, d_out)
        q = self.q.expand(x.size(0), -1, -1) # (B, S_out, d_out)

        y, _ = self.attn(q, x, x, need_weights=False)  # (B, S_out, d_out)
        y = self.norm(y + q)  # residual anchor to preserve slot identity
        return y


class PreVACLTokenUnifier(nn.Module):
    """
    Adapter producing VACL-ready tensors with SAME S_out and DIFFERENT modality dims.

    Inputs:
      - video_feat_3d: (B, C_v_in, T', H', W')
      - audio_tokens:  (B, L,      C_a_in)

    Outputs:
      - X_v: (B, d_v, S_out)
      - X_a: (B, d_a, S_out)
      - aux: dict (debug)
    """
    def __init__(
        self,
        c_v_in: int,  # Swin3D channel dim (e.g., 256)
        c_a_in: int,  # Swin2D token dim (e.g., 768)
        cfg: Optional[TokenUnifierForVACLConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg or TokenUnifierForVACLConfig()

        # You asked NOT to equate modality dims.
        # That means: d_v and d_a may differ.
        #
        # But: if you want shared queries across modalities while keeping d_v != d_a,
        # you cannot literally share the same query tensor (it would have one last-dim).
        #
        # Solution:
        #   - Share a *slot identity* by sharing a query tensor in a latent slot space
        #     and projecting it to each modality's output dim.
        #
        # To keep it simple and still satisfy "shared slots", we:
        #   1) Keep a shared base query in Dq
        #   2) Project base queries into each modality's d_out
        #
        # This preserves shared slot identity without forcing d_v == d_a.
        self._Dq = max(self.cfg.d_v, self.cfg.d_a)  # base query dim (can be any, choose max)

        if self.cfg.share_queries:
            self.base_q = nn.Parameter(torch.randn(1, self.cfg.s_out, self._Dq) * 0.02)
            self.q_to_v = nn.Linear(self._Dq, self.cfg.d_v, bias=False)
            self.q_to_a = nn.Linear(self._Dq, self.cfg.d_a, bias=False)
            shared_q_v = None  # computed dynamically in forward
            shared_q_a = None
        else:
            self.base_q = None
            self.q_to_v = None
            self.q_to_a = None
            shared_q_v = None
            shared_q_a = None

        # Build unifiers (queries injected on forward if sharing)
        self.video_unifier = LearnedQueryUnifier(
            d_in=c_v_in,
            d_out=self.cfg.d_v,
            s_out=self.cfg.s_out,
            n_heads=self.cfg.n_heads,
            attn_dropout=self.cfg.attn_dropout,
            proj_dropout=self.cfg.proj_dropout,
            shared_queries=shared_q_v,  # None here; we'll override if sharing
        )

        self.audio_unifier = LearnedQueryUnifier(
            d_in=c_a_in,
            d_out=self.cfg.d_a,
            s_out=self.cfg.s_out,
            n_heads=self.cfg.n_heads,
            attn_dropout=self.cfg.attn_dropout,
            proj_dropout=self.cfg.proj_dropout,
            shared_queries=shared_q_a,  # None here; we'll override if sharing
        )

    @staticmethod
    def flatten_video(video_feat_3d: torch.Tensor) -> torch.Tensor:
        # (B, C, T', H', W') -> (B, S3D, C)
        if video_feat_3d.ndim != 5:
            raise ValueError(f"[PreVACLTokenUnifier] expected (B,C,T,H,W), got {tuple(video_feat_3d.shape)}")
        B, C, Tp, Hp, Wp = video_feat_3d.shape
        return video_feat_3d.reshape(B, C, Tp * Hp * Wp).transpose(1, 2).contiguous()

    def forward(
        self,
        video_feat_3d: torch.Tensor,  # (B,Cv,T',H',W')
        audio_tokens: torch.Tensor,   # (B,L,Ca)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:

        v_tokens = self.flatten_video(video_feat_3d)  # (B, S3D, Cv)
        a_tokens = audio_tokens
        if a_tokens.ndim != 3:
            raise ValueError(f"[PreVACLTokenUnifier] expected audio (B,L,C), got {tuple(a_tokens.shape)}")

        # If sharing slots, project the shared base queries into each modality output space.
        if self.cfg.share_queries:
            # (1, S_out, Dq) -> (1, S_out, d_v / d_a)
            q_v = self.q_to_v(self.base_q)
            q_a = self.q_to_a(self.base_q)

            # Inject these projected queries into each unifier for this forward pass
            # (we avoid rebuilding modules by temporarily swapping parameters)
            self.video_unifier.q = q_v
            self.audio_unifier.q = q_a

        Zv = self.video_unifier(v_tokens)  # (B, S_out, d_v)
        Za = self.audio_unifier(a_tokens)  # (B, S_out, d_a)

        # VACL expects (B, d, S)
        X_v = Zv.transpose(1, 2).contiguous()  # (B, d_v, S_out)
        X_a = Za.transpose(1, 2).contiguous()  # (B, d_a, S_out)

        aux = {
            "video_in": tuple(video_feat_3d.shape),
            "audio_in": tuple(audio_tokens.shape),
            "S3D_in": int(v_tokens.shape[1]),
            "Sa_in": int(a_tokens.shape[1]),
            "S_out": int(self.cfg.s_out),
            "d_v": int(self.cfg.d_v),
            "d_a": int(self.cfg.d_a),
            "share_queries": bool(self.cfg.share_queries),
        }
        return X_v, X_a, aux


# # Example init (fill in your actual Swin2D token dim Ca_in)
# self.pre_vacl_unifier = PreVACLTokenUnifier(
#     c_v_in=256,      # Swin3D channels from (B,256,T',H',W')
#     c_a_in=768,      # example; set to your Swin2D C from (B,L,C)
#     cfg=TokenUnifierForVACLConfig(
#         s_out=64,
#         d_v=256,      # keep video dim
#         d_a=192,      # example different dim; keep audio dim different if you want
#         n_heads=4,
#         share_queries=True
#     )
# )
#
# # In forward / training step:
# video_feat = self.video_swin3d.forward_features(video_clip)  # (B,Cv,T',H',W')
# audio_tok  = self.audio_swin2d.forward_features(mel)         # (B,L,Ca)
#
# X_v, X_a, aux = self.pre_vacl_unifier(video_feat, audio_tok) # (B,d_v,S), (B,d_a,S)
# loss_vacl = self.vacl(X_v, X_a)