# system_pretrain.py
# ============================================================
# [FINAL PATCHED] AVPretrainSystem (LightningModule)
#
# Responsibilities:
#   - Receives already-collated batches from DataModule
#   - Moves tensors to GPU (Lightning hook)
#   - Runs forward pass
#   - Logs train / val losses to TensorBoard
#   - Optionally tracks energy (CodeCarbon) and FLOPs (fvcore)
#
# IMPORTANT:
#   - No model construction here
#   - No .to(device) inside architecture/backbones
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Iterable, Set, Optional

import torch
import lightning as pl


# ============================================================
# [ADDED] Optional energy tracking (CodeCarbon)
# ============================================================
try:
    from codecarbon import EmissionsTracker  # type: ignore
    _HAS_CODECARBON = True
except Exception:
    EmissionsTracker = None  # type: ignore
    _HAS_CODECARBON = False


# ============================================================
# [ADDED] Optional FLOPs profiling (fvcore)
# ============================================================
try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
    _HAS_FVCORE = True
except Exception:
    FlopCountAnalysis = None  # type: ignore
    _HAS_FVCORE = False


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
        # ============================================================
        enable_energy_tracking: bool = False,
        enable_flops_profile: bool = False,
    ) -> None:
        super().__init__()

        # [KEPT]
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lambda_vacl = float(lambda_vacl)
        self.lambda_cpe = float(lambda_cpe)

        # [ADDED]
        self.enable_energy_tracking = bool(enable_energy_tracking)
        self.enable_flops_profile = bool(enable_flops_profile)
        self._emissions_tracker: Optional[object] = None

        # ============================================================
        # [ADDED] Checkpoint saving config (DDP-safe: rank0 only saves)
        # ============================================================
        self.save_dir = "checkpoints"
        self.save_every_n_epochs = 1
        self.save_weights_only = True
        self._best_val = float("inf")

    # ============================================================
    # [ADDED] Lightning hook: move batch to GPU & normalize
    # ============================================================
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        batch keys:
          - audio          : (B, n_mels, T_audio) float32
          - video_u8_cthw  : (B, 3, T_video, H, W) uint8
        """

        batch["audio"] = batch["audio"].to(device, non_blocking=True)

        batch["video"] = (
            batch["video_u8_cthw"]
            .to(device, non_blocking=True)
            .float()
            .div_(255.0)
        )

        return batch

    # ============================================================
    # [ADDED] Start trackers once at fit start
    # ============================================================
    def on_fit_start(self) -> None:
        if self.enable_energy_tracking and _HAS_CODECARBON and self.global_rank == 0:
            self._emissions_tracker = EmissionsTracker(
                project_name="stage1_ssl",
                output_dir="codecarbon_logs",
                log_level="error",
            )
            self._emissions_tracker.start()

        if self.enable_flops_profile and _HAS_FVCORE and self.global_rank == 0:
            try:
                self._profile_flops_once()
            except Exception as e:
                self.print(f"[WARN] FLOPs profiling failed: {e}")

    # ============================================================
    # [ADDED] Stop trackers at end
    # ============================================================
    def on_fit_end(self) -> None:
        if self.enable_energy_tracking and _HAS_CODECARBON and self._emissions_tracker is not None:
            if self.global_rank == 0:
                emissions = self._emissions_tracker.stop()
                if emissions is not None:
                    self.log(
                        "energy/co2_kg",
                        float(emissions),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=False,
                    )

    # ============================================================
    # [ADDED] FLOPs profiling helper (best-effort)
    # ============================================================
    def _profile_flops_once(self) -> None:
        B, T, H, W = 1, 4, 224, 224

        dummy_video = torch.zeros((B, 3, T, H, W), device=self.device)
        dummy_audio = torch.zeros((B, 1, 64, 96), device=self.device)

        def _forward():
            return self.model(video_in=dummy_video, audio_in=dummy_audio)

        flops = FlopCountAnalysis(_forward, ())
        self.log(
            "profile/flops_per_forward",
            float(flops.total()),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

    # ============================================================
    # [PATCHED] Training step with TensorBoard logging
    # ============================================================
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        audio = batch["audio"].unsqueeze(1)
        video = batch["video"]

        out = self.model(video_in=video, audio_in=audio)

        # [ADDED] Log padding stats occasionally
        if batch_idx % 50 == 0:
            if "Tv_max" in batch:
                self.log("train/Tv_max", float(batch["Tv_max"]), on_step=True, prog_bar=False, sync_dist=True)
            if "Ta_max" in batch:
                self.log("train/Ta_max", float(batch["Ta_max"]), on_step=True, prog_bar=False, sync_dist=True)

        loss_total = out["loss_total"]
        self.log("train/loss_total", loss_total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # [ADDED] Component losses
        if "loss_vacl" in out and out["loss_vacl"] is not None:
            self.log("train/loss_vacl", out["loss_vacl"], on_step=True, on_epoch=True, sync_dist=True)
        if "loss_cpe" in out and out["loss_cpe"] is not None:
            self.log("train/loss_cpe", out["loss_cpe"], on_step=True, on_epoch=True, sync_dist=True)

        return loss_total

    # ============================================================
    # [ADDED] Validation step (SSL stability monitoring)
    # ============================================================
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        audio = batch["audio"].unsqueeze(1)
        video = batch["video"]

        out = self.model(video_in=video, audio_in=audio)

        loss_total = out["loss_total"]
        self.log("val/loss_total", loss_total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if "loss_vacl" in out and out["loss_vacl"] is not None:
            self.log("val/loss_vacl", out["loss_vacl"], on_step=False, on_epoch=True, sync_dist=True)
        if "loss_cpe" in out and out["loss_cpe"] is not None:
            self.log("val/loss_cpe", out["loss_cpe"], on_step=False, on_epoch=True, sync_dist=True)

        return loss_total

    # ============================================================
    # [PATCHED] Explicit optimizer parameter groups (DDP-safe)
    # ============================================================
    def configure_optimizers(self):
        def _ids(params: Iterable[torch.nn.Parameter]) -> Set[int]:
            return {id(p) for p in params}

        vb = getattr(self.model, "video_backbone", None)
        ab = getattr(self.model, "audio_backbone", None)

        vb_params = list(vb.parameters()) if vb is not None else []
        ab_params = list(ab.parameters()) if ab is not None else []

        vb_ids = _ids(vb_params)
        ab_ids = _ids(ab_params)

        other_params = [
            p for p in self.model.parameters()
            if id(p) not in vb_ids and id(p) not in ab_ids
        ]

        param_groups = [
            {"params": [p for p in vb_params if p.requires_grad], "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": [p for p in ab_params if p.requires_grad], "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": [p for p in other_params if p.requires_grad], "lr": self.lr, "weight_decay": self.weight_decay},
        ]

        return torch.optim.AdamW(param_groups)

    # ============================================================
    # [ADDED] Rank-0 checkpoint saver (full .ckpt + optional weights-only)
    # ============================================================
    def _save_checkpoint_files(self, tag: str) -> None:
        """
        Saves:
          - Full Lightning checkpoint: <save_dir>/<tag>.ckpt
          - Optional weights-only:     <save_dir>/<tag>_weights.pt

        DDP-safe: only runs on global_rank == 0.
        """
        if self.global_rank != 0:
            return

        from pathlib import Path

        save_root = Path(self.save_dir)
        save_root.mkdir(parents=True, exist_ok=True)

        # Full Lightning checkpoint (includes optimizer, schedulers, etc.)
        ckpt_path = save_root / f"{tag}.ckpt"
        self.trainer.save_checkpoint(str(ckpt_path))

        # Weights-only payload (small, for inference/extraction)
        if self.save_weights_only:
            weights_path = save_root / f"{tag}_weights.pt"
            payload = {
                "state_dict": self.model.state_dict(),
                "epoch": int(self.current_epoch),
                "global_step": int(self.global_step),
            }
            torch.save(payload, weights_path)

    # ============================================================
    # [ADDED] Save "last" every N epochs
    # ============================================================
    def on_train_epoch_end(self) -> None:
        if (int(self.current_epoch) + 1) % int(self.save_every_n_epochs) == 0:
            self._save_checkpoint_files(tag="last")

    # ============================================================
    # [ADDED] Track best val loss and save "best"
    # ============================================================
    def on_validation_epoch_end(self) -> None:
        metric = self.trainer.callback_metrics.get("val/loss_total", None)
        if metric is None:
            return

        try:
            val_loss = float(metric.detach().cpu())
        except Exception:
            val_loss = float(metric)

        if val_loss < self._best_val:
            self._best_val = val_loss
            self._save_checkpoint_files(tag="best")



