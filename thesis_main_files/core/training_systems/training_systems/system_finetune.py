System pretrain


# system_pretrain.py
# ============================================================
# [FINAL PATCHED] AVPretrainSystem (LightningModule)
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

import torch
import pytorch_lightning as pl

# ============================================================
# [ADDED] Stage-2 classifier head + paper BCE
# ============================================================
from stage2_head import (
    Stage2AVClassifierHead,
    Stage2HeadConfig,
    bce_paper_eq18,
)

# ============================================================
# [ADDED] Metrics (DDP-safe)
# ============================================================
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score


class AVPretrainSystem(pl.LightningModule):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        lambda_vacl: float = 1.0,
        lambda_cpe: float = 1.0,
        # ============================================================
        # [ADDED] Loss weights
        # ============================================================
        omega: float = 1.0,
        lambda_: float = 1.0,
        alpha: float = 0.0,   # L_bmn weight (inactive)
        beta: float = 1.0,    # BCE weight
        # ============================================================
        # [ADDED] Validation memory guards
        # ============================================================
        val_auc_thresholds: int = 256,
        val_metric_cap_batches: int = 200,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------
        # [KEPT] Core config
        # ------------------------------------------------------------
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_vacl = lambda_vacl
        self.lambda_cpe = lambda_cpe

        self.omega = omega
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta

        self.val_auc_thresholds = val_auc_thresholds
        self.val_metric_cap_batches = val_metric_cap_batches

        # ============================================================
        # [ADDED] Freeze Swin backbones (params)
        # ============================================================
        vb = getattr(self.model, "video_backbone", None)
        ab = getattr(self.model, "audio_backbone", None)

        if vb is not None:
            vb.requires_grad_(False)
            vb.eval()

        if ab is not None:
            ab.requires_grad_(False)
            ab.eval()

        # ============================================================
        # [ADDED] Stage-2 head (attached to model – Option B)
        # ============================================================
        if not hasattr(self.model, "stage2_head"):
            k = None
            if hasattr(self.model, "vacl") and hasattr(self.model.vacl, "k"):
                k = int(self.model.vacl.k)
            elif hasattr(self.model, "vacl_wrapper") and hasattr(self.model.vacl_wrapper, "vacl"):
                k = int(self.model.vacl_wrapper.vacl.k)

            if k is None:
                raise RuntimeError("[AVPretrainSystem] Could not infer k for Stage2 head.")

            self.model.stage2_head = Stage2AVClassifierHead(
                k=k,
                cfg=Stage2HeadConfig(num_classes=2),
            )

        # ============================================================
        # [ADDED] Metrics (epoch aggregated, DDP-safe)
        # ============================================================
        self.train_acc = BinaryAccuracy()
        self.train_auc = BinaryAUROC(thresholds=val_auc_thresholds)
        self.train_f1 = BinaryF1Score()

        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC(thresholds=val_auc_thresholds)
        self.val_f1 = BinaryF1Score()

        # ============================================================
        # [ADDED] Inactive stubs (explicitly OFF)
        # ============================================================
        self._USE_LOGITS_LOSS = False
        self._USE_BMN_LOSS = False

        # ============================================================
        # [KEPT] Checkpointing
        # ============================================================
        self.save_dir = Path("checkpoints_stage1_ssl")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._best_val = float("inf")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ============================================================
    # [ADDED] Ensure frozen Swin backbones stay in eval mode
    # ============================================================
    def on_train_epoch_start(self) -> None:
        vb = getattr(self.model, "video_backbone", None)
        ab = getattr(self.model, "audio_backbone", None)
        if vb is not None:
            vb.eval()
        if ab is not None:
            ab.eval()

    # ============================================================
    # [ADDED] Safe batch transfer (labels included)
    # ============================================================
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch["audio"] = batch["audio"].to(device, non_blocking=True)
        batch["video"] = (
            batch["video_u8_cthw"].to(device, non_blocking=True).float().div_(255.0)
        )

        for k in ("y", "label", "y_onehot", "label_onehot"):
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)

        return batch

    # ============================================================
    # [PATCHED] Training step
    # ============================================================
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self.model(
            video_in=batch["video"],
            audio_in=batch["audio"].unsqueeze(1),
        )

        # Required keys
        X_v_att = out["X_v_att"]
        X_a_att = out["X_a_att"]
        L_cor = out["L_cor"]
        l_infonce = out["l_infonce"]

        # Head forward
        head_out = self.model.stage2_head(X_v_att, X_a_att)
        P = head_out["P"]
        logits = head_out["logits"]

        # Labels
        y_onehot = batch.get("y_onehot", batch.get("label_onehot"))
        y_class = batch.get("y", batch.get("label"))

        if y_onehot is None:
            y_class = y_class.long()
            y_onehot = torch.zeros((y_class.size(0), 2), device=y_class.device)
            y_onehot.scatter_(1, y_class[:, None], 1.0)
        else:
            y_onehot = y_onehot.float()
            y_class = y_onehot.argmax(dim=1)

        # ============================================================
        # [ACTIVE] Eq.18 BCE
        # ============================================================
        L_bce = bce_paper_eq18(P, y_onehot)
        L_cls = L_bce

        # ============================================================
        # [OPTIONAL] L_bmn stub (OFF)
        # ============================================================
        L_bmn = None
        if self._USE_BMN_LOSS:
            L_bmn = logits.new_zeros(())

        # ============================================================
        # [FINAL LOSS]
        # ============================================================
        loss_total = (
            self.omega * L_cor
            + self.lambda_ * l_infonce
            + self.beta * L_cls
        )
        if L_bmn is not None:
            loss_total = loss_total + self.alpha * L_bmn

        # ============================================================
        # [LOGS]
        # ============================================================
        self.log("train/loss_total", loss_total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/L_cor", L_cor, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/L_bce", L_bce, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/l_infonce", l_infonce, on_step=True, on_epoch=True, sync_dist=True)

        if L_bmn is not None:
            self.log("train/L_bmn", L_bmn, on_step=True, on_epoch=True, sync_dist=True)

        # Metrics
        score_pos = P[:, 1]
        self.train_acc(score_pos, y_class)
        self.train_auc(score_pos, y_class)
        self.train_f1(score_pos, y_class)

        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/auc", self.train_auc, on_epoch=True, sync_dist=True)
        self.log("train/f1", self.train_f1, on_epoch=True, sync_dist=True)

        return loss_total

    # ============================================================
    # [PATCHED] Validation step (metric cap)
    # ============================================================
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self.model(
            video_in=batch["video"],
            audio_in=batch["audio"].unsqueeze(1),
        )

        X_v_att = out["X_v_att"]
        X_a_att = out["X_a_att"]
        L_cor = out["L_cor"]
        l_infonce = out["l_infonce"]

        head_out = self.model.stage2_head(X_v_att, X_a_att)
        P = head_out["P"]
        logits = head_out["logits"]

        y_onehot = batch.get("y_onehot", batch.get("label_onehot"))
        y_class = batch.get("y", batch.get("label"))

        if y_onehot is None:
            y_class = y_class.long()
            y_onehot = torch.zeros((y_class.size(0), 2), device=y_class.device)
            y_onehot.scatter_(1, y_class[:, None], 1.0)
        else:
            y_onehot = y_onehot.float()
            y_class = y_onehot.argmax(dim=1)

        L_bce = bce_paper_eq18(P, y_onehot)
        L_cls = L_bce

        L_bmn = None
        if self._USE_BMN_LOSS:
            L_bmn = logits.new_zeros(())

        loss_total = (
            self.omega * L_cor
            + self.lambda_ * l_infonce
            + self.beta * L_cls
        )
        if L_bmn is not None:
            loss_total = loss_total + self.alpha * L_bmn

        self.log("val/loss_total", loss_total, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/L_cor", L_cor, on_epoch=True, sync_dist=True)
        self.log("val/L_bce", L_bce, on_epoch=True, sync_dist=True)
        self.log("val/l_infonce", l_infonce, on_epoch=True, sync_dist=True)

        if L_bmn is not None:
            self.log("val/L_bmn", L_bmn, on_epoch=True, sync_dist=True)

        # Metric cap
        if batch_idx < self.val_metric_cap_batches:
            score_pos = P[:, 1]
            self.val_acc(score_pos, y_class)
            self.val_auc(score_pos, y_class)
            self.val_f1(score_pos, y_class)

        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/auc", self.val_auc, on_epoch=True, sync_dist=True)
        self.log("val/f1", self.val_f1, on_epoch=True, sync_dist=True)

        return loss_total

    # ============================================================
    # [KEPT] Optimizer config (frozen Swins excluded automatically)
    # ============================================================
    def configure_optimizers(self):
        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
