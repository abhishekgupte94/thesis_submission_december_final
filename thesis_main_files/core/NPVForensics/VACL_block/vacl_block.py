import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VACLVA(nn.Module):
    """
    VACL module + VA correlation loss for a single modality pair (V, A).

    Implements Eqs. (9)–(16) from the paper, but with only the (V, A)
    branch:

        (9)  M_v = tanh( X_v^T W_jv J_va / sqrt(d_joint) )
        (10) M_a = tanh( X_a^T W_ja J_va / sqrt(d_joint) )
        (11) H_a = ReLU( W_a X_a + W_ma M_a^T )
        (12) H_v = ReLU( W_v X_v + W_mv M_v^T )
        (13) X_v^att = W_hv H_v + X_v
             X_a^att = W_ha H_a + X_a
        (14) X_va   = [ X_v^att ; X_a^att ]
        (15) Loss_va = Σ_i (1 - C_va^ii)^2 + μ Σ_i Σ_{j≠i} (C_va^ij)^2
        (16) L_cor = Loss_va     (since we drop the F–A′ branch)

    Shape conventions (PyTorch):
    -----------------------------
    X_v : (B, d_v, S)
    X_a : (B, d_a, S)

    J_va  : (B, d_joint, S) with d_joint = d_v + d_a
    M_v   : (B, S, S)
    M_a   : (B, S, S)
    H_v   : (B, k, S)
    H_a   : (B, k, S)
    X_va  : (B, d_v + d_a, S)
    """

    def __init__(
        self,
        d_v: int,
        d_a: int,
        seq_len: int,
        k: int,
        mu: float = 0.5,
    ):
        """
        Args
        ----
        d_v     : feature dimension of V (viseme / face)
        d_a     : feature dimension of A (audio)
        seq_len : S, sequence length
        k       : hidden size of attention maps H_v, H_a
        mu      : μ in Eq. (15)
        """
        super().__init__()
        self.d_v = d_v
        self.d_a = d_a
        self.seq_len = seq_len
        self.d_joint = d_v + d_a
        self.k = k
        self.mu = mu

        # Eq. (9)–(10): W_jv ∈ R^{d_v × d_joint}, W_ja ∈ R^{d_a × d_joint}
        self.W_jv = nn.Parameter(torch.Tensor(d_v, self.d_joint))
        self.W_ja = nn.Parameter(torch.Tensor(d_a, self.d_joint))

        # Eq. (11)–(12):
        #   W_a  ∈ R^{k × d_a}, W_v  ∈ R^{k × d_v}
        #   W_ma ∈ R^{k × S},   W_mv ∈ R^{k × S}
        self.W_a = nn.Parameter(torch.Tensor(k, d_a))
        self.W_v = nn.Parameter(torch.Tensor(k, d_v))

        self.W_ma = nn.Parameter(torch.Tensor(k, seq_len))
        self.W_mv = nn.Parameter(torch.Tensor(k, seq_len))

        # Eq. (13):
        #   W_hv ∈ R^{d_v × k}, W_ha ∈ R^{d_a × k}
        self.W_hv = nn.Parameter(torch.Tensor(d_v, k))
        self.W_ha = nn.Parameter(torch.Tensor(d_a, k))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    # ---- Eqs. (9) & (10) ----
    def _compute_joint_matrices(
        self,
        X_v: torch.Tensor,
        X_a: torch.Tensor,
    ):
        """
        X_v : (B, d_v, S)
        X_a : (B, d_a, S)

        Returns:
          J_va : (B, d_joint, S)
          M_v  : (B, S, S)
          M_a  : (B, S, S)
        """
        B, d_v, S = X_v.shape
        _, d_a, S_a = X_a.shape
        assert d_a == self.d_a and S_a == S

        # J_va is concatenation along feature dimension
        J_va = torch.cat([X_v, X_a], dim=1)  # (B, d_v + d_a, S)

        # Eq. (9): M_v
        Xv_T = X_v.transpose(1, 2)              # (B, S, d_v)
        mid_v = torch.matmul(Xv_T, self.W_jv)   # (B, S, d_joint)
        M_v = torch.bmm(mid_v, J_va) / math.sqrt(self.d_joint)
        M_v = torch.tanh(M_v)                   # (B, S, S)

        # Eq. (10): M_a
        Xa_T = X_a.transpose(1, 2)              # (B, S, d_a)
        mid_a = torch.matmul(Xa_T, self.W_ja)   # (B, S, d_joint)
        M_a = torch.bmm(mid_a, J_va) / math.sqrt(self.d_joint)
        M_a = torch.tanh(M_a)                   # (B, S, S)

        return J_va, M_v, M_a

    # ---- Eqs. (11) & (12) ----
    def _compute_attention_maps(
        self,
        X_v: torch.Tensor,
        X_a: torch.Tensor,
        M_v: torch.Tensor,
        M_a: torch.Tensor,
    ):
        """
        Returns:
          H_v : (B, k, S)
          H_a : (B, k, S)
        """
        # Eq. (11) for A
        term_a_feat = torch.einsum("kd,bds->bks", self.W_a, X_a)
        term_a_corr = torch.einsum("ks,bss->bks", self.W_ma,
                                   M_a.transpose(-1, -2))
        H_a = F.relu(term_a_feat + term_a_corr)

        # Eq. (12) for V
        term_v_feat = torch.einsum("kd,bds->bks", self.W_v, X_v)
        term_v_corr = torch.einsum("ks,bss->bks", self.W_mv,
                                   M_v.transpose(-1, -2))
        H_v = F.relu(term_v_feat + term_v_corr)

        return H_v, H_a

    # ---- Eq. (13) & (14) ----
    def _compute_attention_features(
        self,
        X_v: torch.Tensor,
        X_a: torch.Tensor,
        H_v: torch.Tensor,
        H_a: torch.Tensor,
    ):
        """
        Returns:
          X_v_att : (B, d_v, S)
          X_a_att : (B, d_a, S)
          X_va    : (B, d_v + d_a, S)
        """
        delta_v = torch.einsum("dk,bks->bds", self.W_hv, H_v)
        X_v_att = X_v + delta_v

        delta_a = torch.einsum("dk,bks->bds", self.W_ha, H_a)
        X_a_att = X_a + delta_a

        X_va = torch.cat([X_v_att, X_a_att], dim=1)
        return X_v_att, X_a_att, X_va

    # ---- correlation matrix & loss (Eq. 15) ----
    def _correlation_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (B, D, S) -> C ∈ R^{D×D} with entries in [-1, 1].
        """
        B, D, S = X.shape
        N = B * S
        Z = X.permute(0, 2, 1).reshape(N, D)  # (N, D)
        Z = Z - Z.mean(dim=0, keepdim=True)
        Z = Z / (Z.std(dim=0, keepdim=True) + 1e-5)
        C = (Z.T @ Z) / N                     # (D, D)
        return C

    def _correlation_loss_single(self, X: torch.Tensor) -> torch.Tensor:
        """
        Loss_va ≜ Σ_i (1 − C_ii)^2 + μ Σ_i Σ_{j≠i} (C_ij)^2
        """
        C = self._correlation_matrix(X)
        diag = torch.diagonal(C)
        on_diag = (1.0 - diag) ** 2
        off_diag = C - torch.diag(diag)
        return on_diag.sum() + self.mu * (off_diag ** 2).sum()

    # ---- full forward: Eqs. (9)–(16) with only VA ----
    def forward(
        self,
        X_v: torch.Tensor,
        X_a: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          X_va      : fused VA feature (Eq. 14)
          L_cor     : VA correlation loss (Eq. 15 & 16)
          Loss_va   : same as L_cor
          plus intermediates (J_va, M_v, M_a, H_v, H_a, X_v_att, X_a_att)
        """
        J_va, M_v, M_a = self._compute_joint_matrices(X_v, X_a)
        H_v, H_a = self._compute_attention_maps(X_v, X_a, M_v, M_a)
        X_v_att, X_a_att, X_va = self._compute_attention_features(
            X_v, X_a, H_v, H_a
        )

        Loss_va = self._correlation_loss_single(X_va)
        L_cor = Loss_va  # Eq. (16) with only VA branch

        return {
            "X_va": X_va,
            "L_cor": L_cor,
            "Loss_va": Loss_va,
            "J_va": J_va,
            "M_v": M_v,
            "M_a": M_a,
            "H_v": H_v,
            "H_a": H_a,
            "X_v_att": X_v_att,
            "X_a_att": X_a_att,
        }


