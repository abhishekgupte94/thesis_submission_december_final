#!/usr/bin/env python
"""
Sanity check for AVPretrainSystem (system_pretrain.py) with a dummy architecture.

We only test:
- training_step / validation_step mechanics
- optimizer/scheduler wiring
- DDP-safety of logging (on CPU / single GPU)

We do NOT depend on your real Swin or AV wrapper here.
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from core.training_systems.training_systems.system_pretrain import AVPretrainSystem

# [ADDED]
from utils.memory_guard.memory_guard import MemoryGuard


class DummyArchitecture(torch.nn.Module):
    """
    Minimal architecture that matches AVPretrainSystem's expectations:

    forward(batch) -> {
        "module_a_out": {"L_vacl": scalar},
        "module_b_out": {"L_cpe": scalar},
    }
    """

    def __init__(self, d_a: int = 32, d_v: int = 32):
        super().__init__()
        self.lin_a = torch.nn.Linear(d_a, 1)
        self.lin_v = torch.nn.Linear(d_v, 1)

    def forward(self, batch):
        # batch["audio_tokens"]: (B, Sa, D_a)
        # batch["video_tokens"]: (B, Sv, D_v)
        a = batch["audio_tokens"].mean(dim=1)  # (B, D_a)
        v = batch["video_tokens"].mean(dim=1)  # (B, D_v)

        L_vacl = self.lin_a(a).mean()
        L_cpe = self.lin_v(v).mean()

        return {
            "module_a_out": {"L_vacl": L_vacl},
            "module_b_out": {"L_cpe": L_cpe},
        }


class DummySegDataset(Dataset):
    def __init__(self, n_samples=20, Sa=5, Sv=7, d_a=32, d_v=32, guard=None):
        self.n_samples = n_samples
        self.Sa = Sa
        self.Sv = Sv
        self.d_a = d_a
        self.d_v = d_v
        self.guard = guard  # [ADDED]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.guard is not None:
            self.guard.check()  # [ADDED] soft guard during sampling
        return {
            "audio_tokens": torch.randn(self.Sa, self.d_a),
            "video_tokens": torch.randn(self.Sv, self.d_v),
        }


def main():
    # [ADDED] Guards
    guard_strict = MemoryGuard(max_process_gb=8.0, min_system_available_gb=2.0, throws=True)
    guard_soft = MemoryGuard(max_process_gb=8.0, min_system_available_gb=2.0, throws=False)

    guard_strict.check()  # [ADDED]

    d_a = 32
    d_v = 32

    arch = DummyArchitecture(d_a=d_a, d_v=d_v)
    system = AVPretrainSystem(
        architecture=arch,
        lr=1e-3,
        weight_decay=1e-2,
        lambda_vacl=1.0,
        lambda_cpe=1.0,
        use_plateau_scheduler=False,
    )

    guard_soft.check()  # [ADDED] before dataset creation

    train_ds = DummySegDataset(d_a=d_a, d_v=d_v, guard=guard_soft)
    val_ds = DummySegDataset(d_a=d_a, d_v=d_v, guard=guard_soft)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    guard_soft.check()  # [ADDED] before Trainer

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        log_every_n_steps=1,
    )

    trainer.fit(system, train_loader, val_loader)

    guard_soft.check()  # [ADDED] after training


if __name__ == "__main__":
    main()
