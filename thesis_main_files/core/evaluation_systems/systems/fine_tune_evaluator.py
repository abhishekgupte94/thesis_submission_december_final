# core/evaluation_systems/systems/fine_tune_evaluator.py
# ============================================================
# [DROP-IN][PATCHED] AVFineTuneEvaluator (LightningModule)
#
# CHANGE (Option A - NO LAZY HEAD):
#   - Build Stage-2 classifier head eagerly in __init__
#   - Infer head dims from model.cfg (prefers vacl_s_out, fallback to vacl_d_v/d_a)
#   - Remove lazy-init block entirely
#
# Why:
#   - Fixes device mismatch (CPU head vs CUDA activations) by letting Lightning
#     move the head during device placement.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score

from core.training_systems.architectures.final_classifier_module import (
    Stage2AVClassifierHead,
    Stage2HeadConfig,
)


@dataclass
class EvalConfig:
    pool: str = "mean"
    use_layernorm: bool = False
    mlp_hidden: Optional[int] = 128
    dropout: float = 0.1

    # Set True if you want BCE loss too (requires labels present in batch)
    compute_loss: bool = True

    # If set, dumps JSONL predictions (rank-0 only) after predict() finishes
    save_preds_path: Optional[str] = None


class AVFineTuneEvaluator(pl.LightningModule):
    def __init__(self, *, model: nn.Module, cfg: EvalConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg

        # ============================================================
        # [UNCHANGED] Stage-2 head config
        # ============================================================
        self.stage2_cfg = Stage2HeadConfig(
            pool=str(self.cfg.pool),
            use_layernorm=bool(self.cfg.use_layernorm),
            mlp_hidden=self.cfg.mlp_hidden,
            dropout=float(self.cfg.dropout),
            num_classes=1,  # SINGLE logit for BCEWithLogitsLoss
        )

        # ============================================================
        # [PATCHED][OPTION A] Build head eagerly (NO lazy init)
        #   Prefer model.cfg.vacl_s_out (often corresponds to X_*_att dim=K)
        #   Fallback to model.cfg.vacl_d_v / vacl_d_a if available
        # ============================================================
        d_v, d_a = 768,768
        self.stage2_head: Stage2AVClassifierHead = Stage2AVClassifierHead(d_v=d_v, d_a=d_a, cfg=self.stage2_cfg)
        self.stage2_head.eval()

        self._bce = nn.BCEWithLogitsLoss()
        self._pred_rows: List[Dict[str, Any]] = []

        # ============================================================
        # [UNCHANGED] Metrics (per-epoch)
        # ============================================================
        self.val_acc = BinaryAccuracy(threshold=0.5)
        self.val_auc = BinaryAUROC()
        self.val_f1 = BinaryF1Score(threshold=0.5)

        self.test_acc = BinaryAccuracy(threshold=0.5)
        self.test_auc = BinaryAUROC()
        self.test_f1 = BinaryF1Score(threshold=0.5)

        # ============================================================
        # [UNCHANGED] Freeze everything (pure eval)
        # ============================================================
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.stage2_head.parameters():
            p.requires_grad = False

        # (Nice-to-have) keep both in eval mode
        self.model.eval()
        self.stage2_head.eval()

        self.save_hyperparameters(ignore=["model"])

    # ============================================================
    # [ADDED] Head-dim inference helper for Option A
    # ============================================================
    @staticmethod
    def _infer_stage2_head_dims(model: nn.Module) -> Tuple[int, int]:
        """
        Tries to infer the correct Stage-2 head input dims without touching a batch.
        Priority:
          1) model.cfg.vacl_s_out  (common: X_*_att is (B, vacl_s_out, D))
          2) model.cfg.vacl_d_v / model.cfg.vacl_d_a
          3) fallback to 768/768 (last resort)
        """
        cfg = getattr(model, "cfg", None)

        # Prefer vacl_s_out if present (your config sets vacl_s_out=64)
        if cfg is not None and hasattr(cfg, "vacl_s_out"):
            k = int(getattr(cfg, "vacl_s_out"))
            return k, k

        # Fallback: explicit embedding dims
        if cfg is not None and hasattr(cfg, "vacl_d_v") and hasattr(cfg, "vacl_d_a"):
            return int(getattr(cfg, "vacl_d_v")), int(getattr(cfg, "vacl_d_a"))

        # Hard fallback (should not happen in your Stage-2 wiring)
        return 768, 768

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
        return self.model(video_in=batch["video_u8_cthw"], audio_in=batch["audio_96"])

    def _eval_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        out = self.forward(batch)
        X_v_att = out["X_v_att"]
        X_a_att = out["X_a_att"]

        # ============================================================
        # [PATCHED] No lazy head build; head exists and will be on correct device
        # ============================================================
        logits = self.stage2_head(X_v_att=X_v_att, X_a_att=X_a_att)  # (B,1)
        probs = torch.sigmoid(logits)  # (B,1)

        # Labels
        y = batch.get("label", None)
        loss = None

        if self.cfg.compute_loss and y is not None:
            y_f = y.float().view(-1, 1)  # (B,1)
            loss = self._bce(logits, y_f)
            self.log(f"{stage}/bce", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        # Metrics updated per batch, computed/logged per epoch
        if y is not None and stage in ("val", "test"):
            y_bin = y.int().view(-1)  # (B,)
            p = probs.view(-1)  # (B,)

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

        clip_ids = batch.get("clip_id", [None] * len(probs))
        seg_idxs = batch.get("seg_idx", [None] * len(probs))

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
