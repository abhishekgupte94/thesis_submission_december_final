# system_finetune_patched.py
# ============================================================
# [SYSTEM] Fine-Tune Lightning System
#
# NOTE:
# - This file is IDENTICAL to system_finetune.py
# - EXCEPT for fvcore FLOPs profiling + CodeCarbon emissions tracking
# - All additions are clearly marked below
# - No other behavior, logic, or wiring has been modified
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import lightning as pl

# ============================================================
# [ADDED][FLOPS + CARBON] Imports (mirrors pretrain system)
# ============================================================
from fvcore.nn import FlopCountAnalysis
from codecarbon import EmissionsTracker
# ============================================================


class AVFineTuneSystem(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        *,
        profile_flops: bool = False,
        track_carbon: bool = False,
    ):
        super().__init__()

        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.profile_flops = profile_flops
        self.track_carbon = track_carbon

        # ============================================================
        # [ADDED][FLOPS + CARBON] Runtime state (mirrors pretrain)
        # ============================================================
        self._flops_profiled: bool = False
        self._emissions_tracker: Optional[EmissionsTracker] = None
        # ============================================================

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------
    # Training Step
    # ------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        out = self.model(**batch)

        loss = out["loss"]
        self.log("train/loss", loss, prog_bar=True)

        return loss

    # ------------------------------------------------------------
    # Validation Step
    # ------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = out["loss"]

        self.log("val/loss", loss, prog_bar=True)

        return loss

    # ------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            **self.optimizer_cfg,
        )

        if self.scheduler_cfg is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            **self.scheduler_cfg,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    # ============================================================
    # [ADDED][FLOPS + CARBON] Lightning lifecycle hooks
    # ============================================================
    def on_fit_start(self):
        """
        Mirrors pretrain system:
        - Start CodeCarbon tracker (rank 0 only)
        """
        if self.track_carbon and self.global_rank == 0:
            self._emissions_tracker = EmissionsTracker(
                project_name="fine_tune_training",
                log_level="warning",
            )
            self._emissions_tracker.start()

    def on_fit_end(self):
        """
        Mirrors pretrain system:
        - Stop CodeCarbon tracker (rank 0 only)
        """
        if self._emissions_tracker is not None:
            self._emissions_tracker.stop()
            self._emissions_tracker = None
    # ============================================================

    # ============================================================
    # [ADDED][FLOPS] One-time FLOPs profiling helper
    # ============================================================
    def _profile_flops_once(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Identical behavior to pretrain system:
        - Runs once
        - Rank 0 only
        - Uses fvcore FlopCountAnalysis
        """
        if self._flops_profiled:
            return

        if self.global_rank != 0:
            return

        try:
            self.model.eval()
            with torch.no_grad():
                flops = FlopCountAnalysis(self.model, batch)
                total_flops = flops.total()

            self.log(
                "profile/flops",
                float(total_flops),
                rank_zero_only=True,
            )

        except Exception as e:
            print(f"[WARN] FLOPs profiling failed: {e}")

        finally:
            self._flops_profiled = True
            self.model.train()
    # ============================================================

    # ------------------------------------------------------------
    # Hook used during training to trigger FLOPs profiling
    # ------------------------------------------------------------
    def on_train_batch_start(self, batch, batch_idx):
        if self.profile_flops:
            self._profile_flops_once(batch)
