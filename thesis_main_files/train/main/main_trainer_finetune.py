# thesis_main_files/train/train_stage2_finetune.py
# ============================================================
# [FINAL MAIN TRAIN SCRIPT - STAGE 2 FINETUNE]
# Lightning 2.x entrypoint.
#
# Builds:
#   - SegmentDataModule (must yield labels y or y_onehot)
#   - Swin2D (frozen)
#   - Swin3D (frozen)
#   - Stage-2 finetune architecture (must output X_v_att, X_a_att, L_cor, l_infonce)
#   - AVPretrainSystem from system_fine.py
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.dataloaders.dataloader import SegmentDataModule

# ============================================================
# [MODIFIED] Import renamed system file
# ============================================================
from core.training_systems.training_systems.system_finetune import AVPretrainSystem

from scripts.feature_extraction.SWIN.main.MAIN_swin2d_wrapper import (
    Swin2DAudioBackboneWrapper,
    Swin2DAudioWrapperConfig,
)
from scripts.feature_extraction.SWIN.main.MAIN_swin_3d_wrapper import VideoBackboneSwin3D
from scripts.feature_extraction.SWIN.main.build_swin3d import BuildSwin3DConfig

# ============================================================
# [NOTE] You must point this to your real finetune architecture module
# It must return: X_v_att, X_a_att, L_cor, l_infonce
# ============================================================
from core.training_systems.architectures.vacl_finetune_arch import VACLFinetuneArchitecture


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--offline-root", type=str, required=True)
    p.add_argument("--batch-name", type=str, required=True)

    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--bucket-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)

    p.add_argument("--precision", type=str, default="bf16-mixed",
                   choices=["32-true", "16-mixed", "bf16-mixed"])

    p.add_argument("--val-split", type=float, default=0.05)

    # Loss weights
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--lambda_", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=1.0)

    # Val memory guards
    p.add_argument("--val-auc-thresholds", type=int, default=256)
    p.add_argument("--val-metric-cap-batches", type=int, default=200)

    p.add_argument("--tb-logdir", type=str, default="tb_logs")
    p.add_argument("--run-name", type=str, default="stage2_finetune")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    offline_root = (REPO_ROOT / args.offline_root).resolve()
    tb_logdir = (REPO_ROOT / args.tb_logdir).resolve()
    tb_logdir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=str(tb_logdir), name=args.run_name)

    # DataModule (must yield y or y_onehot)
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

    # Swin2D (audio) - frozen
    audio_backbone = Swin2DAudioBackboneWrapper(Swin2DAudioWrapperConfig())
    audio_backbone.requires_grad_(False)
    audio_backbone.eval()

    # Swin3D (video) - frozen
    swin3d_cfg = BuildSwin3DConfig(out="5d", use_checkpoint=True)
    video_backbone = VideoBackboneSwin3D(swin3d_cfg)
    video_backbone.requires_grad_(False)
    video_backbone.eval()

    # Stage-2 finetune architecture
    model = VACLFinetuneArchitecture(
        video_backbone=video_backbone,
        audio_backbone=audio_backbone,
    )

    system = AVPretrainSystem(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        omega=args.omega,
        lambda_=args.lambda_,
        alpha=args.alpha,
        beta=args.beta,
        val_auc_thresholds=args.val_auc_thresholds,
        val_metric_cap_batches=args.val_metric_cap_batches,
    )

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
