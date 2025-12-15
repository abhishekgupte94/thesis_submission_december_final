#!/usr/bin/env python
"""
Sanity check for VACLVA (vacl_block.py)

- Creates random V/A features with correct shapes.
- Runs forward() to get X_va and L_cor.
- Runs backward() to confirm gradients flow and loss is finite.
"""

import torch
from core.NPVForensics.VACL_block.vacl_block import VACLVA

# [ADDED]
from utils.memory_guard.memory_guard import MemoryGuard


def main():
    # [ADDED] Strict guard before we allocate anything big
    guard_strict = MemoryGuard(max_process_gb=8.0, min_system_available_gb=2.0, throws=True)
    guard_soft = MemoryGuard(max_process_gb=8.0, min_system_available_gb=2.0, throws=False)

    guard_strict.check()  # [ADDED]

    # Hyperparams (keep small for sanity)
    B = 4       # batch size
    d_v = 64    # video feature dim
    d_a = 64    # audio feature dim
    S = 10      # sequence length
    k = 32      # hidden size inside VACL
    mu = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VACLVA(
        d_v=d_v,
        d_a=d_a,
        seq_len=S,
        k=k,
        mu=mu,
    ).to(device)

    guard_soft.check()  # [ADDED] before allocating input tensors

    # Fake inputs
    X_v = torch.randn(B, d_v, S, device=device, requires_grad=True)
    X_a = torch.randn(B, d_a, S, device=device, requires_grad=True)

    guard_soft.check()  # [ADDED] before forward

    out = model(X_v, X_a)
    X_va = out["X_va"]
    L_cor = out["L_cor"]

    print("X_va shape:", X_va.shape)  # expect (B, d_v + d_a, S)
    print("L_cor:", L_cor.item())

    assert X_va.shape == (B, d_v + d_a, S), "Unexpected X_va shape"
    assert torch.isfinite(L_cor), "L_cor has NaN/Inf"

    guard_soft.check()  # [ADDED] before backward

    # Backward
    L_cor.backward()
    grad_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
    print("Total grad norm:", grad_norm)

    guard_soft.check()  # [ADDED] after backward


if __name__ == "__main__":
    main()
