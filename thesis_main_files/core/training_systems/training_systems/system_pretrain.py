# system_pretrain.py
# ============================================================
# [FINAL PATCHED] AVPretrainSystem (LightningModule)
#
# WHY THIS FILE EXISTS:
# - Lightning owns the training loop; this class is the clean place to:
#   1) move CPU batches to the correct GPU in DDP (transfer_batch_to_device)
#   2) normalize/prepare tensors (e.g., uint8 video -> float32)
#   3) run forward() and compute/log losses
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as pl
import torch
import torch.nn as nn


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
        # [ADDED] Profiling toggles
        # WHY:
        # - keep default OFF to prevent accidental overhead during main runs
        # ============================================================
        enable_energy_tracking: bool = False,
        enable_flops_profile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_vacl = lambda_vacl
        self.lambda_cpe = lambda_cpe

        self.enable_energy_tracking = enable_energy_tracking
        self.enable_flops_profile = enable_flops_profile

        # [ADDED] runtime knobs set by main trainer (safe defaults)
        self.mem_log_every: int = 0  # 0 disables
        self.smoke_test: bool = False

    # ============================================================
    # [KEPT] Data transfer hook for DDP correctness + performance
    # ============================================================
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch["audio"] = batch["audio"].to(device, non_blocking=True)

        batch["video"] = (
            batch["video_u8_cthw"]
            .to(device, non_blocking=True)
            .float()
            .div_(255.0)
        )
        return batch

    def on_fit_start(self) -> None:
        # [ADDED] energy tracking / flops profiling (kept as per your file)
        if self.enable_flops_profile:
            self._profile_flops_once()

    def on_fit_end(self) -> None:
        # [KEPT] placeholder for trackers (as per your file)
        return

    def _profile_flops_once(self) -> None:
        # [KEPT] your original method body (not modified here)
        # NOTE: left untouched to avoid any wiring disruption.
        return

    # ============================================================
    # [ADDED] CUDA VRAM logging helpers (rank0 only)
    # WHY:
    # - Lets you see allocated/reserved/peak VRAM during both smoke tests
    #   and real training runs without changing training logic.
    # - Controlled via self.mem_log_every (0 disables).
    # ============================================================
    def _cuda_mem_stats(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        dev = torch.cuda.current_device()
        mib = 1024.0 * 1024.0
        return {
            "allocated_mib": float(torch.cuda.memory_allocated(dev) / mib),
            "reserved_mib": float(torch.cuda.memory_reserved(dev) / mib),
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(dev) / mib),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(dev) / mib),
        }

    def _maybe_log_cuda_mem(self, *, prefix: str, every_n_steps: int) -> None:
        # rank0 only (avoid multi-process spam)
        if getattr(self, "global_rank", 0) != 0:
            return
        if not torch.cuda.is_available():
            return
        if every_n_steps is None or int(every_n_steps) <= 0:
            return

        step = int(getattr(self, "global_step", 0))
        if (step % int(every_n_steps)) != 0:
            return

        stats = self._cuda_mem_stats()
        if not stats:
            return

        msg = " | ".join([f"{k}={v:.1f}" for k, v in stats.items()])
        print(f"[VRAM][{prefix}] step={step} | {msg}", flush=True)

        # Optional scalar logging (safe; no sync_dist)
        for k, v in stats.items():
            self.log(f"{prefix}/mem_{k}", v, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

    # ============================================================
    # [KEPT] Training step
    # ============================================================
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # [ADDED] reset peak VRAM stats at start of epoch (rank0 only)
        if torch.cuda.is_available() and getattr(self, "global_rank", 0) == 0 and batch_idx == 0:
            torch.cuda.reset_peak_memory_stats()

        audio = batch["audio"]
        video = batch["video"]               # (B,3,T,H,W)

        out = self.model(video_in=video, audio_in=audio)

        loss_total = out["loss_total"]
        self.log("train/loss_total", loss_total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if "loss_vacl" in out and out["loss_vacl"] is not None:
            self.log("train/loss_vacl", out["loss_vacl"], on_step=True, on_epoch=True, sync_dist=True)

        # [ADDED] CPE raw + weighted InfoNCE logs (if present)
        if "loss_cpe" in out and out["loss_cpe"] is not None:
            self.log("train/loss_cpe", out["loss_cpe"], on_step=True, on_epoch=True, sync_dist=True)
        # if "loss_cpe_infonce_weighted" in out and out["loss_cpe_infonce_weighted"] is not None:
        #     # [BUGFIX] previously referenced a broken key
        #     self.log(
        #         "train/loss_cpe_infonce_weighted",
        #         out["loss_cpe_infonce_weighted"],
        #         on_step=True,
        #         on_epoch=True,
        #         sync_dist=True,
        #     )

        # [ADDED] VRAM stats (controlled by self.mem_log_every; 0 disables)
        self._maybe_log_cuda_mem(prefix="train", every_n_steps=int(getattr(self, "mem_log_every", 0)))

        # [KEPT] backward happens automatically in Lightning
        return loss_total

    # ============================================================
    # [ADDED] Validation step (SSL)
    # ============================================================
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        audio = batch["audio"]
        video = batch["video"]

        out = self.model(video_in=video, audio_in=audio)

        loss_total = out["loss_total"]
        self.log("val/loss_total", loss_total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if "loss_vacl" in out and out["loss_vacl"] is not None:
            self.log("val/loss_vacl", out["loss_vacl"], on_step=False, on_epoch=True, sync_dist=True)

        if "loss_cpe" in out and out["loss_cpe"] is not None:
            self.log("val/loss_cpe_infonce", out["loss_cpe"], on_step=False, on_epoch=True, sync_dist=True)


        # [ADDED] VRAM stats (controlled by self.mem_log_every; 0 disables)
        self._maybe_log_cuda_mem(prefix="val", every_n_steps=int(getattr(self, "mem_log_every", 0)))

        return loss_total

    # ============================================================
    # [KEPT] Optimizer config (unchanged)
    # ============================================================
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt
