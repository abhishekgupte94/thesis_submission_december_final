#!/usr/bin/env python
"""
Sanity check for CPE/common-space modules:

- CommonSpaceProjector
- MultiModalProjectionHeads
- FaceAudioCommonSpaceWrapper

We test:
- (B, D) embeddings
- (B, T, D) sequence embeddings
- InfoNCE loss in both cases
"""

import torch

from core.NPVForensics.common_projection.common_space_projector import CommonSpaceProjector
from core.NPVForensics.common_projection.multimodal_projection_heads import MultiModalProjectionHeads
from core.NPVForensics.common_projection.main.common_projection_head_module_wrapper import FaceAudioCommonSpaceWrapper

# [ADDED]
from utils.memory_guard.memory_guard import MemoryGuard


def test_common_space_projector(guard_strict: MemoryGuard, guard_soft: MemoryGuard):
    print("\n=== CommonSpaceProjector sanity ===")

    guard_strict.check()  # [ADDED]

    B, T, D_in, D_out = 4, 5, 64, 32

    proj_1 = CommonSpaceProjector(in_dim=D_in, out_dim=D_out, num_layers=1)
    proj_2 = CommonSpaceProjector(in_dim=D_in, out_dim=D_out, num_layers=2)

    guard_soft.check()  # [ADDED] before big tensors

    x_2d = torch.randn(B, D_in)
    x_3d = torch.randn(B, T, D_in)

    guard_soft.check()  # [ADDED] before forwards

    y_2d_1 = proj_1(x_2d)
    y_3d_1 = proj_1(x_3d)

    y_2d_2 = proj_2(x_2d)
    y_3d_2 = proj_2(x_3d)

    print("y_2d_1 shape:", y_2d_1.shape)
    print("y_3d_1 shape:", y_3d_1.shape)
    print("y_2d_2 shape:", y_2d_2.shape)
    print("y_3d_2 shape:", y_3d_2.shape)

    guard_soft.check()  # [ADDED]


def test_multimodal_projection_heads(guard_strict: MemoryGuard, guard_soft: MemoryGuard):
    print("\n=== MultiModalProjectionHeads sanity ===")

    guard_strict.check()  # [ADDED]

    B, T = 4, 6
    d_f, d_a, d_fa = 64, 48, 32

    X_f = torch.randn(B, T, d_f)
    X_a = torch.randn(B, T, d_a)

    guard_soft.check()  # [ADDED] before module

    heads = MultiModalProjectionHeads(d_a=d_a, d_f=d_f, d_fa=d_fa)
    out = heads(X_f=X_f, X_a=X_a)

    print("Z_f_fa shape:", out["Z_f_fa"].shape)  # (B, T, d_fa)
    print("Z_a_fa shape:", out["Z_a_fa"].shape)  # (B, T, d_fa)

    guard_soft.check()  # [ADDED]


def test_face_audio_wrapper(guard_strict: MemoryGuard, guard_soft: MemoryGuard):
    print("\n=== FaceAudioCommonSpaceWrapper sanity ===")

    guard_strict.check()  # [ADDED]

    B, T = 4, 6
    d_f, d_a, d_fa = 64, 48, 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_f = torch.randn(B, T, d_f, device=device, requires_grad=True)
    X_a = torch.randn(B, T, d_a, device=device, requires_grad=True)

    guard_soft.check()  # [ADDED] before module

    wrapper = FaceAudioCommonSpaceWrapper(
        d_a=d_a,
        d_f=d_f,
        d_fa=d_fa,
        temperature=0.1,
        loss_weight=1.0,
    ).to(device)

    guard_soft.check()  # [ADDED] before forward

    out = wrapper(X_f=X_f, X_a=X_a, compute_ec_loss=True)

    Z_f_fa = out["Z_f_fa"]
    Z_a_fa = out["Z_a_fa"]
    L_info = out["L_info"]

    print("Z_f_fa shape:", Z_f_fa.shape)
    print("Z_a_fa shape:", Z_a_fa.shape)
    print("L_info:", float(L_info))

    assert torch.isfinite(L_info), "L_info is NaN/Inf"

    guard_soft.check()  # [ADDED] before backward

    # Backward
    loss = L_info + (Z_f_fa.mean() + Z_a_fa.mean())
    loss.backward()
    print("Backward OK (gradients exist)")

    guard_soft.check()  # [ADDED] after backward


def main():
    # [ADDED] Global guards for this script
    guard_strict = MemoryGuard(max_process_gb=8.0, min_system_available_gb=2.0, throws=True)
    guard_soft = MemoryGuard(max_process_gb=8.0, min_system_available_gb=2.0, throws=False)

    test_common_space_projector(guard_strict, guard_soft)
    test_multimodal_projection_heads(guard_strict, guard_soft)
    test_face_audio_wrapper(guard_strict, guard_soft)


if __name__ == "__main__":
    main()
