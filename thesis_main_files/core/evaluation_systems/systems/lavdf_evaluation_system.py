# core/boundary_localisation/main/lavdf_eval_system_tb.py
# ============================================================
# [FINAL PATCHED] LAV-DF Lightning Evaluator System + TensorBoard
#
# Features:
#   - Evaluation-only LightningModule
#   - Logs metrics to TensorBoard
#   - Optional CSV output (rank-0)
#   - Optional JSONL output (rank-0)  <-- ADDED
#
# Assumes batch contains:
#   batch["video_paths"] : List[str]
#   batch["y"]           : Tensor[B]
#   optional batch["clip_ids"]
# ============================================================

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
)


# ============================================================
# Config
# ============================================================
@dataclass
class EvalSystemConfig:
    threshold: float = 0.5
    save_preds_path: Optional[str] = None    # CSV
    save_jsonl_path: Optional[str] = None    # JSONL (NEW)
    log_prob_hist: bool = True


# ============================================================
# LightningModule
# ============================================================
class LAVDFEvalSystem(pl.LightningModule):
    """
    Evaluation-only system.
    """

    def __init__(
        self,
        *,
        arch: nn.Module,
        cfg: Optional[EvalSystemConfig] = None,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.cfg = EvalSystemConfig() if cfg is None else cfg

        # Metrics
        self.val_acc = BinaryAccuracy(threshold=self.cfg.threshold)
        self.val_f1 = BinaryF1Score(threshold=self.cfg.threshold)
        self.val_auroc = BinaryAUROC()

        # Buffers
        self._csv_rows: List[List[Any]] = []
        self._jsonl_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    def configure_optimizers(self):
        return None

    # ------------------------------------------------------------
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        out = self.arch(batch)

        if "prob_fake" not in out:
            raise KeyError("Architecture output missing 'prob_fake'")

        prob = out["prob_fake"]                  # (B,)
        y = batch["y"].to(prob.device).long()    # (B,)
        logits2 = out.get("logits2", None)

        # Update metrics
        self.val_acc.update(prob, y)
        self.val_f1.update(prob, y)
        self.val_auroc.update(prob, y)

        clip_ids = batch.get(
            "clip_ids",
            [f"sample_{batch_idx}_{i}" for i in range(len(prob))]
        )
        video_paths = batch.get("video_paths", [None] * len(prob))

        # Collect outputs
        for i in range(len(prob)):
            row_csv = [
                str(clip_ids[i]),
                int(y[i].item()),
                float(prob[i].item()),
                int(prob[i].item() >= self.cfg.threshold),
            ]
            self._csv_rows.append(row_csv)

            seg_idxs = batch.get("seg_idxs", [0] * len(prob))

            for i in range(len(prob)):
                row_json = {
                    "clip_id": str(clip_ids[i]),
                    "seg_idx": int(seg_idxs[i]),  # <-- ADDED
                    "video_path": video_paths[i],
                    "label": int(y[i].item()),
                    "prob_fake": float(prob[i].item()),
                    "pred@0.5": int(prob[i].item() >= self.cfg.threshold),
                }
                if logits2 is not None:
                    row_json["logits"] = logits2[i].detach().cpu().tolist()

                self._jsonl_rows.append(row_json)

        # TensorBoard histogram
        if self.cfg.log_prob_hist and self.trainer.is_global_zero:
            self.logger.experiment.add_histogram(
                "val/prob_fake",
                prob.detach().float().cpu(),
                global_step=self.global_step,
            )

    # ------------------------------------------------------------
    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        auroc = self.val_auroc.compute()

        self.log("val/acc@0.5", acc, prog_bar=True, sync_dist=True)
        self.log("val/f1@0.5", f1, prog_bar=True, sync_dist=True)
        self.log("val/auroc", auroc, prog_bar=True, sync_dist=True)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()

        # Write CSV
        if self.cfg.save_preds_path is not None and self.trainer.is_global_zero:
            out_path = Path(self.cfg.save_preds_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "label", "prob_fake", "pred@0.5"])
                w.writerows(self._csv_rows)
            self._csv_rows.clear()

        # Write JSONL
        if self.cfg.save_jsonl_path is not None and self.trainer.is_global_zero:
            out_path = Path(self.cfg.save_jsonl_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                for row in self._jsonl_rows:
                    f.write(json.dumps(row) + "\n")
            self._jsonl_rows.clear()
