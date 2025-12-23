# core/training_systems/training_systems/system_fine_eval.py
# ============================================================
# [DROP-IN] AVFineTuneEvaluator (LightningModule)
#
# Purpose:
#   - Loads Stage-2 architecture (your "model") + classifier head
#   - Runs evaluation only (no optimizer, no training_step)
#   - Computes per-epoch metrics:
#       * Accuracy@0.5
#       * AUROC
#       * F1@0.5
#   - Optionally saves per-sample predictions (rank-0 only)
#
# Notes:
#   - DDP/Lightning-friendly (sync_dist=True)
#   - No device moves inside your model; Lightning handles batch->device
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as pl
import torch
import torch.nn as nn

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score

from core.training_systems.architectures.final_classifier_module import Stage2AVClassifierHead


@dataclass
class EvalConfig:
    pool: str = "mean"
    use_layernorm: bool = False
    mlp_hidden: Optional[int] = None
    dropout: float = 0.0

    # Set True if you want BCE loss too (requires labels present in batch)
    compute_loss: bool = True

    # If set, dumps JSONL predictions (rank-0 only) after predict() finishes
    save_preds_path: Optional[str] = None


class AVFineTuneEvaluator(pl.LightningModule):
    def __init__(self, *, model: nn.Module, cfg: EvalConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg

        self.stage2_head = Stage2AVClassifierHead(
            pool=str(cfg.pool),
            use_layernorm=bool(cfg.use_layernorm),
            mlp_hidden=cfg.mlp_hidden,
            dropout=float(cfg.dropout),
        )

        self._bce = nn.BCEWithLogitsLoss()
        self._pred_rows: List[Dict[str, Any]] = []

        # ============================================================
        # [ADDED] Metrics (per-epoch)
        # ============================================================
        self.val_acc = BinaryAccuracy(threshold=0.5)
        self.val_auc = BinaryAUROC()
        self.val_f1 = BinaryF1Score(threshold=0.5)

        self.test_acc = BinaryAccuracy(threshold=0.5)
        self.test_auc = BinaryAUROC()
        self.test_f1 = BinaryF1Score(threshold=0.5)

        # Freeze everything by default (pure eval)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.stage2_head.parameters():
            p.requires_grad = False

        self.save_hyperparameters(ignore=["model", "stage2_head"])

    # Keep same behavior as your fine-tune system: move tensors, keep strings/ints as-is
    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        if isinstance(batch, dict):
            out: Dict[str, Any] = {}
            for k, v in batch.items():
                out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            return out
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Expect your model signature: model(video_in=..., audio_in=...)
        return self.model(video_in=batch["video"], audio_in=batch["audio"])

    def _eval_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        out = self.forward(batch)
        X_v_att = out["X_v_att"]
        X_a_att = out["X_a_att"]

        logits = self.stage2_head(X_v_att=X_v_att, X_a_att=X_a_att)  # (B,1)
        probs = torch.sigmoid(logits)  # (B,1)

        y = batch.get("label", None)
        loss = None

        if self.cfg.compute_loss and y is not None:
            y_f = y.float().view(-1, 1)
            loss = self._bce(logits, y_f)
            self.log(f"{stage}/bce", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        # ============================================================
        # [ADDED] Metrics updated per batch, computed/logged per epoch
        # ============================================================
        if y is not None:
            # torchmetrics expects:
            #   probs: (B,) float in [0,1]
            #   target: (B,) int {0,1}
            y_bin = y.int().view(-1)
            p = probs.view(-1)

            if stage == "val":
                self.val_acc.update(p, y_bin)
                self.val_auc.update(p, y_bin)
                self.val_f1.update(p, y_bin)

                self.log("val/acc@0.5", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log("val/f1@0.5", self.val_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

            elif stage == "test":
                self.test_acc.update(p, y_bin)
                self.test_auc.update(p, y_bin)
                self.test_f1.update(p, y_bin)

                self.log("test/acc@0.5", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log("test/f1@0.5", self.test_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return {"logits": logits, "probs": probs, "y": y, "loss": loss}

    # ---------------------------
    # Lightning loop hooks
    # ---------------------------
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._eval_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._eval_step(batch, stage="test")

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        out = self._eval_step(batch, stage="pred")
        return {
            "probs": out["probs"].detach().cpu(),
            "logits": out["logits"].detach().cpu(),
        }

    # ---------------------------
    # Metric resets (per-epoch)
    # ---------------------------
    def on_validation_epoch_end(self) -> None:
        self.val_acc.reset()
        self.val_auc.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self) -> None:
        self.test_acc.reset()
        self.test_auc.reset()
        self.test_f1.reset()

    # ---------------------------
    # Optional prediction dumping
    # ---------------------------
    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        if self.cfg.save_preds_path is None:
            return
        if not self.trainer.is_global_zero:
            return

        probs = outputs["probs"].view(-1).tolist()
        logits = outputs["logits"].view(-1).tolist()

        labels = batch.get("label", None)
        labels_list = labels.view(-1).tolist() if labels is not None else [None] * len(probs)

        # If your batch provides ids, include them; otherwise fill Nones
        clip_ids = batch.get("clip_id", [None] * len(probs))
        seg_idxs = batch.get("seg_idx", [None] * len(probs))

        # clip_ids / seg_idxs might be tensors; coerce if needed
        if torch.is_tensor(clip_ids):
            clip_ids = clip_ids.detach().cpu().tolist()
        if torch.is_tensor(seg_idxs):
            seg_idxs = seg_idxs.detach().cpu().tolist()

        for i in range(len(probs)):
            self._pred_rows.append(
                {
                    "clip_id": clip_ids[i] if i < len(clip_ids) else None,
                    "seg_idx": seg_idxs[i] if i < len(seg_idxs) else None,
                    "prob": probs[i],
                    "logit": logits[i],
                    "label": labels_list[i],
                }
            )

    def on_predict_end(self) -> None:
        if self.cfg.save_preds_path is None:
            return
        if not self.trainer.is_global_zero:
            return

        import json

        p = Path(self.cfg.save_preds_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for row in self._pred_rows:
                f.write(json.dumps(row) + "\n")

        print(f"[AVFineTuneEvaluator] Wrote predictions: {p}")

    # ---------------------------
    # No optimizer (pure evaluator)
    # ---------------------------
    def configure_optimizers(self):
        return None
