# system_pretrain.py

"""
Lightning system wrapper for AVPretrainArchitecture.

This module defines AVPretrainSystem, a pl.LightningModule that:
    - wraps AVPretrainArchitecture
    - defines training/validation steps
    - configures optimizers/schedulers

IMPORTANT:
    - Sections marked "KEEP YOUR ORIGINAL IMPLEMENTATION HERE" are
      placeholders where you should paste your existing logic.
    - Changes are marked with [ADDED] / [MODIFIED].
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
import lightning

from core.training_systems.architectures.pretrain_architecture import AVPretrainArchitecture, ArchitectureConfig


class AVPretrainSystem(pl.LightningModule):
    """
    LightningModule that *trains* the AVPretrainArchitecture using
    VACL + CPE/EC losses.

    Expected usage:
        - Construct an AVPretrainArchitecture externally
        - Wrap it in AVPretrainSystem
        - Pass this system to pl.Trainer(...).fit()

    NOTE:
        The architecture is pure nn.Module (no training logic inside).
        All training behaviour lives here (loss combination, logging,
        optimizers, schedulers).
    """

    def __init__(
        self,
        architecture: AVPretrainArchitecture,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        lambda_vacl: float = 1.0,
        lambda_cpe: float = 1.0,
        use_plateau_scheduler: bool = True,
    ) -> None:
        super().__init__()

        # Store the wrapped architecture
        self.arch = architecture

        # Optim configuration
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lambda_vacl = float(lambda_vacl)
        self.lambda_cpe = float(lambda_cpe)
        self.use_plateau_scheduler = bool(use_plateau_scheduler)

        # This is already Lightning-idiomatic (B1.9)
        self.save_hyperparameters(ignore=["architecture"])

        # --------------------------------------------------------------
        # KEEP ANY EXTRA METRIC OBJECTS OR STATE YOU ALREADY HAD
        # (e.g. torchmetrics.Accuracy, EMA trackers, etc.)
        # --------------------------------------------------------------

    # --------------------------------------------------------------
    # Forward just delegates to the architecture  (B1.10)
    # --------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass used for:
            - training_step / validation_step
            - optional inference / feature extraction

        Simply delegates to the underlying architecture.
        """
        return self.arch(batch)

    # --------------------------------------------------------------
    # Loss extraction & combination
    # --------------------------------------------------------------
    def _extract_losses(self, out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Extract and combine loss components from architecture outputs.

        This function should:
          - Read loss terms from `module_a_out` and `module_b_out`
          - Apply lambda_vacl / lambda_cpe weights
          - Return all as tensors (no .item() here)

        Replace the example implementation body with your existing one
        if it already does this.
        """
        module_a_out = out["module_a_out"]
        module_b_out = out["module_b_out"]

        # We assume your VACL head exposes a scalar loss like "L_vacl"
        L_vacl = module_a_out.get("L_vacl")
        if L_vacl is None:
            raise KeyError("module_a_out must contain key 'L_vacl'")

        # And your CPE/EC head exposes a scalar loss like "L_cpe" or "L_ec"
        L_cpe = module_b_out.get("L_cpe") or module_b_out.get("L_ec")
        if L_cpe is None:
            raise KeyError("module_b_out must contain key 'L_cpe' or 'L_ec'")

        # Weighted total loss (matches your original lambda_vacl/cpe usage)
        L_total = self.lambda_vacl * L_vacl + self.lambda_cpe * L_cpe

        return {
            "L_vacl": L_vacl,
            "L_cpe": L_cpe,
            "L_total": L_total,
        }

    # --------------------------------------------------------------
    # [ADDED] Centralised loss combiner (B2.11)
    # --------------------------------------------------------------
    def compute_total_loss(self, out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Thin wrapper around `_extract_losses` so that
        training/validation steps stay minimal and we have a single
        place where the final scalar `L_total` is defined.
        """
        return self._extract_losses(out)

    # --------------------------------------------------------------
    # Training / validation steps (B2.12, B2.13)
    # --------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(batch)
        loss_dict = self.compute_total_loss(out)  # [MODIFIED]
        loss = loss_dict["L_total"]

        # DDP-safe logging with sync_dist=True
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,  # [ADDED]
        )
        self.log(
            "train/L_vacl",
            loss_dict["L_vacl"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,  # [ADDED]
        )
        self.log(
            "train/L_cpe",
            loss_dict["L_cpe"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,  # [ADDED]
        )

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(batch)
        loss_dict = self.compute_total_loss(out)  # [MODIFIED]
        loss = loss_dict["L_total"]

        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,  # [ADDED]
        )
        self.log(
            "val/L_vacl",
            loss_dict["L_vacl"],
            on_epoch=True,
            sync_dist=True,  # [ADDED]
        )
        self.log(
            "val/L_cpe",
            loss_dict["L_cpe"],
            on_epoch=True,
            sync_dist=True,  # [ADDED]
        )

        return loss

    # --------------------------------------------------------------
    # [MODIFIED] Optimizer + optional LR scheduler (B3.14)
    # --------------------------------------------------------------
    def configure_optimizers(self):
        """
        Advanced optimizer configuration with AdamW and
        optional ReduceLROnPlateau scheduler.

        - Separates Swin backbone params from the rest (param groups).
        - Currently uses the same learning rate for both groups, but
          this makes it very easy to adjust them separately later.
        """
        backbone_params = []  # [ADDED]
        head_params = []      # [ADDED]

        for name, p in self.named_parameters():  # [ADDED]
            if not p.requires_grad:
                continue
            if "swin" in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params},  # [ADDED]
                {"params": head_params},      # [ADDED]
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.use_plateau_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                },
            }

        # If scheduler is disabled, just return the optimizer
        return optimizer

    # --------------------------------------------------------------
    # KEEP YOUR EXISTING FLOP PROFILING / CALLBACK HOOKS HERE
    # (if you already had a profile_flops() or similar)
    # --------------------------------------------------------------
