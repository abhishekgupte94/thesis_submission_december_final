# thesis_main_files/train/train_stage1.py
# ============================================================
# [FINAL MAIN TRAIN SCRIPT]
#
# WHY THIS FILE EXISTS:
# - One entry point that wires everything together:
#   1) Resolve repo-relative paths (prevents saving under wrong cwd)
#   2) Create DataModule (train/val split + bucketing)
#   3) Build backbones (Swin2D + Swin3D wrappers)
#   4) Build architecture (unifier + VACL + CPE; returns scalar losses)
#   5) Build Lightning system (logs + saving + DDP-safe transfers)
#   6) Launch Trainer (DDP, precision, TensorBoard)
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger


# ============================================================
# [ADDED] Repo root anchor
# WHY:
# - ensures any relative paths are interpreted under thesis_main_files/
# - fixes your earlier issue where passing "data/processed" created nested paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# [PROJECT IMPORTS]
# ============================================================
from scripts.dataloaders.dataloader import SegmentDataModule
from core.training_systems.architectures.pretrain_architecture import AVPretrainArchitecture, ArchitectureConfig
from core.training_systems.training_systems.system_pretrain import AVPretrainSystem

from scripts.feature_extraction.SWIN.main.MAIN_swin2d_wrapper import Swin2DAudioBackboneWrapper, Swin2DAudioWrapperConfig
from scripts.feature_extraction.SWIN.main.MAIN_swin_3d_wrapper import VideoBackboneSwin3D
from scripts.feature_extraction.SWIN.main.build_swin3d import BuildSwin3DConfig


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

    # precision (A100: bf16-mixed is recommended)
    p.add_argument("--precision", type=str, default="bf16-mixed",
                   choices=["32-true", "16-mixed", "bf16-mixed"])

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

    # ============================================================
    # [ADDED] matmul precision hint
    # WHY:
    # - improves throughput on modern GPUs without changing model code
    # ============================================================
    torch.set_float32_matmul_precision("high")

    # ============================================================
    # [ADDED] Resolve repo-relative paths
    # ============================================================
    offline_root = (REPO_ROOT / args.offline_root).resolve()
    tb_logdir = (REPO_ROOT / args.tb_logdir).resolve()
    tb_logdir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # [ADDED] TensorBoard logger
    # WHY:
    # - Lightning self.log(...) writes scalars here automatically
    # - produces train/val curves
    # ============================================================
    logger = TensorBoardLogger(
        save_dir=str(tb_logdir),
        name=args.run_name,
    )

    # ============================================================
    # DataModule (val_split=0.05 + bucketing by T_video)
    # WHY:
    # - bucketing reduces padding => lower VRAM spikes + better throughput
    # - val_split yields stable SSL validation curves
    # ============================================================
    dm = SegmentDataModule(
        offline_root=offline_root,
        batch_name=args.batch_name,
        batch_size=args.batch_size,
        bucket_size=args.bucket_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        map_location="cpu",
        seed=123,
        val_split=float(args.val_split),
        val_batch_size=args.batch_size,
    )

    # ============================================================
    # Build Swin2D (audio) wrapper
    # ============================================================
    audio_backbone = Swin2DAudioBackboneWrapper(
        Swin2DAudioWrapperConfig(
            # keep wrapper defaults unless you need overrides
        )
    )

    # ============================================================
    # Build Swin3D (video) wrapper
    # ============================================================
    swin3d_cfg = BuildSwin3DConfig(
        out="5d",
        use_checkpoint=True,  # activation checkpointing reduces VRAM
    )
    video_backbone = VideoBackboneSwin3D(swin3d_cfg)

    # ============================================================
    # Build Architecture (includes dedicated CPE InfoNCE weight)
    # ============================================================
    arch_cfg = ArchitectureConfig(
        vacl_s_out=64,
        vacl_d_v=256,
        vacl_d_a=768,
        compute_infonce=True,
        return_intermediates=False,
        lambda_vacl=1.0,
        lambda_cpe_infonce=float(args.lambda_cpe_infonce),  # [ADDED]
    )

    model = AVPretrainArchitecture(
        cfg=arch_cfg,
        video_backbone=video_backbone,
        audio_backbone=audio_backbone,
        c_v_in=256,
        c_a_in=768,
    )

    # ============================================================
    # Lightning System (logs + saving + DDP transfers)
    # ============================================================
    system = AVPretrainSystem(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_vacl=1.0,
        lambda_cpe=1.0,
        enable_energy_tracking=args.enable_energy_tracking,
        enable_flops_profile=args.enable_flops_profile,
    )

    # ============================================================
    # [ADDED] Configure checkpoint saving under thesis_main_files/
    # ============================================================
    system.save_dir = str((REPO_ROOT / "checkpoints").resolve())
    system.save_every_n_epochs = 1
    system.save_weights_only = True

    # ============================================================
    # Trainer (DDP + precision + TensorBoard)
    # WHY:
    # - DDP: 8 GPUs
    # - precision: bf16-mixed recommended for A100 training stability/speed
    # - check_val_every_n_epoch: ensures val curve exists
    # ============================================================
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp",
        precision=args.precision,
        max_epochs=args.max_epochs,
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    trainer.fit(system, datamodule=dm)


if __name__ == "__main__":
    main()


