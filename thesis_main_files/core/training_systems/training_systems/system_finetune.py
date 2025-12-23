# core/training_systems/training_systems/system_fine.py
# ============================================================
# [FINAL PATCHED] AVFineTuneSystem (LightningModule)
#
# Purpose:
#   Stage-2 finetune Lightning system that:
#     - runs the Stage-2 finetune architecture (pure nn.Module)
#     - computes losses (VACL correlation + InfoNCE/CPE + BCE classification)
#     - trains a binary classifier head on discriminative features:
#         X_v_att and X_a_att
#
# Contract (REQUIRED from model forward):
#   forward(...) returns dict with:
#       "X_v_att":   (B, k, S)
#       "X_a_att":   (B, k, S)
#       "L_cor":     scalar
#       "l_infonce": scalar  (REQUIRED)
#
# Notes:
#   - DDP/Lightning-friendly
#   - No device moves in the model; system handles batch->device
#   - bf16-mixed safe
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# [KEPT] Stage-2 head (binary classifier)
# ============================================================
from core.training_systems.architectures.final_classifier_module import Stage2AVClassifierHead, Stage2HeadConfig


@dataclass
class AVFineTuneSystemConfig:
    # Loss weights (names preserved)
    omega: float = 1.0     # weight for correlation/VACL loss
    lambda_: float = 0.1   # weight for InfoNCE/CPE loss
    alpha: float = 1.0     # weight for BCE classification loss
    beta: float = 0.0      # optional extra term (kept for compatibility)

    # Stage-2 head defaults
    head_pool: str = "mean"          # "mean" or "max"
    head_use_layernorm: bool = False
    head_mlp_hidden: Optional[int] = None
    head_dropout: float = 0.0

    # Validation knobs
    val_auc_thresholds: Tuple[float, ...] = (0.5,)
    val_metric_cap_batches: int = 0  # 0 => no cap

    # Optional profiling toggles
    enable_energy_tracking: bool = False
    enable_flops_profile: bool = False


class AVFineTuneSystem(pl.LightningModule):
    def __init__(
        self,
        *,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        omega: float = 1.0,
        lambda_vacl: float = 1.0,   # kept to avoid breaking older callers
        lambda_cpe: float = 0.1,    # kept to avoid breaking older callers
        alpha: float = 1.0,
        beta: float = 0.0,
        stage2_pool: str = "mean",
        stage2_use_layernorm: bool = False,
        stage2_mlp_hidden: Optional[int] = None,
        stage2_dropout: float = 0.0,
        lr_head: Optional[float] = None,
        weight_decay_head: Optional[float] = None,
        lr_backbone: Optional[float] = None,
        weight_decay_backbone: Optional[float] = None,
        val_auc_thresholds: Tuple[float, ...] = (0.5,),
        val_metric_cap_batches: int = 0,
        enable_energy_tracking: bool = False,
        enable_flops_profile: bool = False,
    ) -> None:
        super().__init__()

        # ============================================================
        # [KEPT] Core fields
        # ============================================================
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        # ============================================================
        # [KEPT] Loss weights (preserve naming for compatibility)
        # ============================================================
        self.omega = float(omega)
        self.lambda_vacl = float(lambda_vacl)
        self.lambda_cpe = float(lambda_cpe)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # ============================================================
        # [KEPT] Stage-2 head
        # ============================================================
        # ============================================================
        # [PATCH] Stage2 head construction – FIXED kwargs mismatch
        # ============================================================

        self.stage2_cfg = Stage2HeadConfig(
            pool=str(stage2_pool),
            use_layernorm=bool(stage2_use_layernorm),
            mlp_hidden=stage2_mlp_hidden,
            dropout=float(stage2_dropout),
            num_classes=1,  # SINGLE logit for BCEWithLogitsLoss
        )

        self.stage2_head = Stage2AVClassifierHead(
            d_v=self.d_v,  # or whatever variable holds video feature dim
            d_a=self.d_a,  # or whatever variable holds audio feature dim
            cfg=self.stage2_cfg
        )

        # ============================================================
        # [KEPT] Optional param-group overrides
        # ============================================================
        self.lr_head = lr_head
        self.weight_decay_head = weight_decay_head
        self.lr_backbone = lr_backbone
        self.weight_decay_backbone = weight_decay_backbone

        # ============================================================
        # [KEPT] Validation knobs
        # ============================================================
        self.val_auc_thresholds = tuple(val_auc_thresholds)
        self.val_metric_cap_batches = int(val_metric_cap_batches)

        # ============================================================
        # [KEPT] Profiling toggles
        # ============================================================
        self.enable_energy_tracking = bool(enable_energy_tracking)
        self.enable_flops_profile = bool(enable_flops_profile)

        # ============================================================
        # [MIRRORED] Runtime knobs (set by trainer script)
        # ============================================================
        self.mem_log_every: int = 50
        self.smoke_test: bool = False

        # Loss for binary classification
        self._bce = nn.BCEWithLogitsLoss()

        # Lightning: save hparams (safe: excludes full model)
        self.save_hyperparameters(
            ignore=["model", "stage2_head"],
        )
        # ============================================================
        # [ADDED] Freeze audio + video backbones for Stage-2 finetune
        # ============================================================
        self._freeze_backbones()
    # ============================================================
    # [ADDED] Backbone freezing (Stage-2 policy)
    # ============================================================
    def _freeze_backbones(self) -> None:
        """
        Freeze audio + video backbones.
        Stage-2 trains ONLY:
          - VACL
          - CPE / InfoNCE
          - Stage-2 classifier head
        """
        frozen = 0

        if hasattr(self.model, "video_backbone"):
            for p in self.model.video_backbone.parameters():
                p.requires_grad = False
                frozen += 1

        if hasattr(self.model, "audio_backbone"):
            for p in self.model.audio_backbone.parameters():
                p.requires_grad = False
                frozen += 1

        # Optional: log once on rank 0
        if self.trainer is None or self.trainer.is_global_zero:
            print(f"[AVFineTuneSystem] Frozen backbone parameters: {frozen}")

    # ============================================================
    # [KEPT] Batch transfer hook (DDP-safe)
    # ============================================================
    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        if isinstance(batch, dict):
            out = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    out[k] = v.to(device, non_blocking=True)
                else:
                    out[k] = v
            return out
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    # ============================================================
    # [KEPT] Forward: calls pure model and returns its dict
    # ============================================================
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Expect dataloader to provide:
        #   batch["video"] : tensor
        #   batch["audio"] : tensor
        # Optional:
        #   batch["label"] : (B,) 0/1 or float
        return self.model(
            video_in=batch["video"],
            audio_in=batch["audio"],
        )

    # ============================================================
    # [KEPT] Optimizer with optional param groups
    # ============================================================
    def configure_optimizers(self):
        # Default: one param group
        params = [{"params": self.parameters(), "lr": self.lr, "weight_decay": self.weight_decay}]

        # If you want separate groups, you need stable access to head/backbone params.
        # This code keeps compatibility and only activates when overrides are supplied.
        if (
            self.lr_head is not None
            or self.weight_decay_head is not None
            or self.lr_backbone is not None
            or self.weight_decay_backbone is not None
        ):
            head_lr = self.lr if self.lr_head is None else float(self.lr_head)
            head_wd = self.weight_decay if self.weight_decay_head is None else float(self.weight_decay_head)
            bb_lr = self.lr if self.lr_backbone is None else float(self.lr_backbone)
            bb_wd = self.weight_decay if self.weight_decay_backbone is None else float(self.weight_decay_backbone)

            # Split params into: head vs rest
            head_params = list(self.stage2_head.parameters())
            head_param_ids = {id(p) for p in head_params}

            backbone_params = []
            other_params = []
            for p in self.model.parameters():
                (backbone_params if True else other_params).append(p)  # kept simple; model decides freeze

            # Replace with 2 groups: head + everything else
            params = [
                {"params": backbone_params + other_params, "lr": bb_lr, "weight_decay": bb_wd},
                {"params": head_params, "lr": head_lr, "weight_decay": head_wd},
            ]

        opt = torch.optim.AdamW(params)
        return opt

    # ============================================================
    # [KEPT] Shared step: compute losses + logits
    # ============================================================
    def _shared_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        out = self.forward(batch)

        # Required keys from architecture
        X_v_att = out["X_v_att"]
        X_a_att = out["X_a_att"]
        L_cor = out["L_cor"]
        l_infonce = out["l_infonce"]

        # Stage-2 logits from head
        logits = self.stage2_head(X_v_att=X_v_att, X_a_att=X_a_att)  # (B,1)

        # Labels
        y = batch.get("label", None)
        if y is None:
            # Allow pure forward/profiling without labels
            loss_cls = torch.zeros((), device=logits.device, dtype=logits.dtype)
        else:
            # BCEWithLogitsLoss expects float targets of shape (B,1)
            y = y.float().view(-1, 1)
            loss_cls = self._bce(logits, y)

        # Weighted total loss
        loss = (
            self.omega * L_cor
            + self.lambda_cpe * l_infonce
            + self.alpha * loss_cls
        )

        # Logs
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_cor", L_cor, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_infonce", l_infonce, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_cls", loss_cls, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        return {
            "loss": loss,
            "logits": logits.detach(),
            "y": None if y is None else y.detach(),
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._shared_step(batch, stage="val")


