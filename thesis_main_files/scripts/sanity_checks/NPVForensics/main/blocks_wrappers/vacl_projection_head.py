#!/usr/bin/env python
"""
Sanity check for VACLProjectionHead (vacl_wrapper.py)

- Tests both "bsd" and "bds" input layouts.
- Confirms projection output shape and scalar loss.
"""

import torch
from core.NPVForensics.VACL_block.main.vacl_wrapper import VACLProjectionHead

# [ADDED]
from utils.memory_guard.memory_guard import MemoryGuard


def run_case(input_layout: str, guard_strict: MemoryGuard, guard_soft: MemoryGuard):
    print(f"\n=== Testing VACLProjectionHead with input_layout='{input_layout}' ===")

    guard_strict.check()  # [ADDED] at the start of each case

    B = 4
    S = 10
    d_v = 64
    d_a = 64
    k = 32
    out_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head = VACLProjectionHead(
        d_v=d_v,
        d_a=d_a,
        seq_len=S,
        k=k,
        out_dim=out_dim,
        mu=0.5,
        input_layout=input_layout,
        pool="mean",
    ).to(device)

    guard_soft.check()  # [ADDED] before inputs

    if input_layout == "bds":
        X_v = torch.randn(B, d_v, S, device=device, requires_grad=True)
        X_a = torch.randn(B, d_a, S, device=device, requires_grad=True)
    else:  # "bsd"
        X_v = torch.randn(B, S, d_v, device=device, requires_grad=True)
        X_a = torch.randn(B, S, d_a, device=device, requires_grad=True)

    guard_soft.check()  # [ADDED] before forward

    out = head(X_v, X_a, return_dict=True)
    proj = out["proj"]
    L_cor = out["L_cor"]

    print("proj shape:", proj.shape)  # expect (B, out_dim)
    print("L_cor:", float(L_cor))

    assert proj.shape == (B, out_dim)
    assert torch.isfinite(L_cor), "L_cor is NaN/Inf"

    guard_soft.check()  # [ADDED] before backward

    # Backward
    loss = L_cor + proj.mean()
    loss.backward()
    print("Backward OK (gradients exist)")

    guard_soft.check()  # [ADDED] after backward


def main():
    # [ADDED] Shared guards for both cases
    guard_strict = MemoryGuard(max_process_gb=8.0, min_system_available_gb=2.0, throws=True)
    guard_soft = MemoryGuard(max_process_gb=8.0, min_system_available_gb=2.0, throws=False)

    run_case("bds", guard_strict, guard_soft)
    run_case("bsd", guard_strict, guard_soft)


if __name__ == "__main__":
    main()
