# thesis_main_files/train/train_stage1.py
# ============================================================
# [FINAL MAIN TRAIN SCRIPT]
#
# - Anchors paths to thesis_main_files/ (REPO_ROOT)
# - Uses TensorBoardLogger for train/val curves
# - Uses SegmentDataModule with val_split=0.05
# - Builds:
#     Swin3D wrapper (video)
#     Swin2D wrapper (audio)
#     AVPretrainArchitecture
#     AVPretrainSystem
# - Runs Lightning Trainer with DDP
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger


# ============================================================
# [ADDED] Repo root anchor:
# this file lives at thesis_main_files/train/train_stage1.py
# so parents[1] == thesis_main_files/
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# [EXISTING IMPORTS] Your project modules
# (Adjust import paths only if your repo differs)
# ============================================================
from scripts.dataloaders.dataloader import SegmentDataModule  # patched dataloader with val_split
from core.training_systems.architectures.pretrain_architecture  import AVPretrainArchitecture, ArchitectureConfig
from core.training_systems.training_systems.system_pretrain import AVPretrainSystem

from scripts.feature_extraction.SWIN.main.MAIN_swin2d_wrapper import Swin2DAudioBackboneWrapper, Swin2DAudioWrapperConfig
from scripts.feature_extraction.SWIN.main.MAIN_swin_3d_wrapper import VideoBackboneSwin3D
from scripts.feature_extraction.SWIN.main.build_swin3d import BuildSwin3DConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # ------------------------------------------------------------
    # Data / paths (all are RELATIVE to thesis_main_files/)
    # ------------------------------------------------------------
    p.add_argument("--offline-root", type=str, required=True,
                   help="Path under thesis_main_files/ to offline tensors root (e.g. data/processed/.../AVSpeech_offline_training_files)")
    p.add_argument("--batch-name", type=str, required=True,
                   help="Name of the batch folder under offline-root (e.g. avspeech_video_stage1)")

    # ------------------------------------------------------------
    # Training knobs
    # ------------------------------------------------------------
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size")
    p.add_argument("--bucket-size", type=int, default=8, help="Bucket width in frames for T_video bucketing")
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)

    # ------------------------------------------------------------
    # Precision (A100 recommended: bf16-mixed)
    # ------------------------------------------------------------
    p.add_argument("--precision", type=str, default="bf16-mixed",
                   choices=["32-true", "16-mixed", "bf16-mixed"])

    # ------------------------------------------------------------
    # Validation split (we lock this to 0.05 as requested,
    # but still keep it configurable for experiments)
    # ------------------------------------------------------------
    p.add_argument("--val-split", type=float, default=0.05)

    # ------------------------------------------------------------
    # Optional profiling toggles
    # ------------------------------------------------------------
    p.add_argument("--enable-energy-tracking", action="store_true",
                   help="Enable CodeCarbon tracking (rank0 only). Requires: pip install codecarbon")
    p.add_argument("--enable-flops-profile", action="store_true",
                   help="Enable fvcore FLOPs profile (rank0 only). Requires: pip install fvcore")

    # ------------------------------------------------------------
    # Logging / output
    # ------------------------------------------------------------
    p.add_argument("--tb-logdir", type=str, default="tb_logs",
                   help="TensorBoard log dir under thesis_main_files/")
    p.add_argument("--run-name", type=str, default="stage1_ssl",
                   help="TensorBoard run name")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ============================================================
    # [ADDED] A100 performance hint
    # ============================================================
    torch.set_float32_matmul_precision("high")

    # ============================================================
    # [ADDED] Resolve repo-relative paths safely
    # ============================================================
    offline_root = (REPO_ROOT / args.offline_root).resolve()
    tb_logdir = (REPO_ROOT / args.tb_logdir).resolve()
    tb_logdir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # [ADDED] TensorBoard logger
    # (Lightning will write train/* and val/* from self.log calls)
    # ============================================================
    logger = TensorBoardLogger(
        save_dir=str(tb_logdir),
        name=args.run_name,
    )

    # ============================================================
    # [PATCHED] DataModule (val_split=0.05)
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
        val_split=float(args.val_split),  # <- 0.05 per your request
        val_batch_size=args.batch_size,   # same per-GPU size for val
    )

    # ============================================================
    # [PATCHED] Build backbones (wrappers call external repos)
    # ============================================================

    # -------------------
    # Swin2D (audio)
    # -------------------
    audio_backbone = Swin2DAudioBackboneWrapper(
        Swin2DAudioWrapperConfig(
            # Keep defaults from wrapper unless you need overrides
        )
    )

    # -------------------
    # Swin3D (video)
    # -------------------
    swin3d_cfg = BuildSwin3DConfig(
        # IMPORTANT: these fields must match your BuildSwin3DConfig dataclass.
        # Fill any required fields that your builder expects.
        out="5d",
        use_checkpoint=True,  # [RECOMMENDED] big activation memory saver
        # pretrained=..., pretrained2d=..., etc if required by your builder
    )
    video_backbone = VideoBackboneSwin3D(swin3d_cfg)

    # ============================================================
    # [PATCHED] Build architecture
    # ============================================================
    arch_cfg = ArchitectureConfig(
        vacl_s_out=64,
        vacl_d_v=256,
        vacl_d_a=768,
        compute_infonce=True,
        return_intermediates=False,
        lambda_vacl=1.0,
        lambda_cpe=1.0,
    )

    model = AVPretrainArchitecture(
        cfg=arch_cfg,
        video_backbone=video_backbone,
        audio_backbone=audio_backbone,
        c_v_in=256,
        c_a_in=768,
    )

    # ============================================================
    # [PATCHED] Lightning system (train/val logging inside)
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
    # [ADDED] Configure checkpoint saving location under thesis_main_files/
    # ============================================================
    system.save_dir = str((REPO_ROOT / "checkpoints").resolve())
    system.save_every_n_epochs = 1
    system.save_weights_only = True

    # ============================================================
    # [ADDED] Trainer
    # ============================================================
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp",
        precision=args.precision,
        max_epochs=args.max_epochs,
        logger=logger,
        check_val_every_n_epoch=1,   # [ADDED] ensures val loss is produced
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    trainer.fit(system, datamodule=dm)


if __name__ == "__main__":
    main()

