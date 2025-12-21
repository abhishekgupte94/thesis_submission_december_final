# thesis_main_files/train/train_stage1.py
# ============================================================
# [FINAL MAIN TRAIN SCRIPT]
#
# WHY THIS FILE EXISTS:
# - One entry point that wires everything together:
#   1) Resolve repo-relative paths (prevents saving under wrong cwd)
#   2) Create DataModule (train/val split + bucketing)
#   3) Build backbones (Swin2D + Swin3D wrappers)
#   4) Build architecture (AVPretrainArchitecture)
#   5) Build Lightning System (AVPretrainSystem)
#   6) Build Trainer (DDP + bf16-mixed)
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

# NOTE: keep your existing imports below as-is in your repo:
from  scripts.dataloaders.dataloader import SegmentDataModule
from core.training_systems.architectures.pretrain_architecture import AVPretrainArchitecture
from core.training_systems.training_systems.system_pretrain import AVPretrainSystem
from scripts.feature_extraction.SWIN.wrapper.main_wrapper_swins import build_backbones_for_training

REPO_ROOT = Path(__file__).resolve().parents[2]  # adjust if your original file differs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # data / paths (relative to thesis_main_files/)
    p.add_argument("--offline-root", type=str, required=True)
    p.add_argument("--batch-name", type=str, required=True)

    # training knobs
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--bucket-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)

    # [ADDED] Gradient accumulation to preserve effective batch without VRAM increase
    p.add_argument("--accumulate-grad-batches", type=int, default=1)

    # [ADDED] CUDA VRAM print frequency (0 disables)
    p.add_argument("--mem-log-every", type=int, default=50)

    # [ADDED] 1-GPU smoke test mode (tiny run + no checkpointing)
    p.add_argument("--smoke-test", action="store_true")

    # precision (A100: bf16-mixed is recommended)
    p.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
    )

    # validation split (locked to your requested default)
    p.add_argument("--val-split", type=float, default=0.05)

    # profiling toggles
    p.add_argument("--enable-energy-tracking", action="store_true")
    p.add_argument("--enable-flops-profile", action="store_true")

    # tensorboard logging
    p.add_argument("--tb-logdir", type=str, default="tb_logs")
    p.add_argument("--run-name", type=str, default="stage1_ssl")

    # ============================================================
    # [ADDED] CPE InfoNCE weight (your new control knob)
    # WHY:
    # - lets you tune the influence of CPE InfoNCE on total SSL objective
    # ============================================================
    p.add_argument("--lambda-cpe-infonce", type=float, default=1.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    offline_root = (
        str((REPO_ROOT / args.offline_root).resolve())
        if not Path(args.offline_root).is_absolute()
        else args.offline_root
    )

    # ============================================================
    # DataModule (KEPT)
    # ============================================================
    dm = SegmentDataModule(
        offline_root=offline_root,
        batch_name=args.batch_name,
        batch_size=args.batch_size,
        bucket_size=args.bucket_size,
        num_workers=args.num_workers,
        # pin_memory=True,
        # persistent_workers=True,
        val_split=args.val_split,
    )

    # ============================================================
    # Build backbones + architecture (KEPT)
    # ============================================================
    video_backbone,audio_backbone = build_backbones_for_training()

    model = AVPretrainArchitecture(
        audio_backbone=audio_backbone,
        video_backbone=video_backbone
        # lambda_cpe_infonce=args.lambda_cpe_infonce,
    )

    # ============================================================
    # Lightning System (KEPT)
    # ============================================================
    system = AVPretrainSystem(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_vacl=1.0,
        lambda_cpe=0.1,
        enable_energy_tracking=args.enable_energy_tracking,
        enable_flops_profile=args.enable_flops_profile,
    )

    # ============================================================
    # [ADDED] Runtime knobs (do not affect wiring)
    # - mem_log_every: VRAM print frequency (0 disables)
    # - smoke_test: lets the System print VRAM every step if you want
    # ============================================================
    system.mem_log_every = int(args.mem_log_every)
    system.smoke_test = bool(args.smoke_test)

    logger = TensorBoardLogger(
        save_dir=str((REPO_ROOT / args.tb_logdir).resolve()),
        name=args.run_name,
    )

    # ============================================================
    # Trainer (DDP + precision + TensorBoard)
    # WHY:
    # - DDP: 8 GPUs (default)
    # - precision: bf16-mixed recommended for A100 training stability/speed
    # - check_val_every_n_epoch: ensures val curve exists
    # ============================================================

    # ============================================================
    # [ADDED] Smoke-test overrides (1 GPU, tiny batches, no ckpt)
    # ============================================================
    if args.smoke_test:
        devices = 1
        strategy = "auto"
        max_epochs = 1
        limit_train_batches = 2
        limit_val_batches = 1
        num_sanity_val_steps = 0
        enable_checkpointing = False
        log_every_n_steps = 1
        # make VRAM output immediately useful
        system.mem_log_every = 1
    else:
        devices = args.devices
        strategy = "ddp"
        max_epochs = args.max_epochs
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        num_sanity_val_steps = 2
        enable_checkpointing = True
        log_every_n_steps = 10

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=args.precision,
        max_epochs=max_epochs,
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=enable_checkpointing,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=num_sanity_val_steps,
    )

    trainer.fit(system, datamodule=dm)


if __name__ == "__main__":
    main()
