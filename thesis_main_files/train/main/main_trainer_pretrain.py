#!/usr/bin/env python
"""
train_av_pretrain.py
====================

Main training script for the AV pretraining system:

    - Uses AVSegmentTokenDataModule to load **tokenised** segments from offline .pt files.
    - Builds the architecture:
        * Swin backbone (video/audio feature extractor)
        * Positional encoder / pre-architecture wrapper
        * VACLProjectionHead (Module A)
        * FaceAudioCommonSpaceWrapper (Module B / EC head)
    - Wraps everything in AVPretrainSystem (LightningModule).
    - Trains with PyTorch Lightning on single- or multi-GPU (e.g., 8× A100).

You will likely need to adjust the following:

    1) Imports for your Swin backbone + AV wrapper (see "MODEL IMPORTS" section).
    2) Feature dims (d_v, d_a, seq_len, etc.) in the config block.
    3) Index JSON path and data root.

The rest is plug-and-play.
"""

import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# -------------------------------
# MODEL IMPORTS (ADJUST THESE)
# -------------------------------

# Core architecture + Lightning system
from core.training_systems.architectures.pretrain_architecture import AVPretrainArchitecture, ArchitectureConfig
from core.training_systems.training_systems.system_pretrain import AVPretrainSystem

# VACL + CPE modules
from core.NPVForensics.VACL_block.main.vacl_wrapper import VACLProjectionHead
from core.NPVForensics.common_projection.main.common_projection_head_module_wrapper import FaceAudioCommonSpaceWrapper

# DataModule for tokenised AV segments
from scripts.dataloaders.dataloader import AVSegmentTokenDataModule

# TODO [YOU]: adjust these imports to match your actual Swin/Positional wrapper:
# Example possibilities (you will know the real names):
# from swin_backbone_wrapper import build_swin_backbone
# from main_preprocessing_block import AVWrapper
#
# For now, we'll assume you have:
#   - a function `build_swin_backbone(args)` that returns a nn.Module
#   - an AV wrapper class `MainAVWrapper` that takes no args or some args
#
# Replace the lines below with your actual implementations.
from typing import Tuple
import torch.nn as nn


def build_swin_backbone_stub(d_model: int) -> nn.Module:
    """
    [TODO YOU] Replace this stub with your actual Swin backbone builder.

    It should:
        - Take configuration (e.g. d_model, patch size, depth, etc.)
        - Return an nn.Module that consumes the batch dict prepared by your
          AV wrapper and outputs modality-specific features needed by VACL/CPE.

    For now this is a dummy identity module so the script is syntactically valid.
    """
    class IdentityBackbone(nn.Module):
        def forward(self, batch):
            # Expect that batch already has "audio_tokens" and "video_tokens"
            # shaped as (B, Sa, D_a) and (B, Sv, D_v) and simply return them.
            return {
                "audio_features": batch["audio_tokens"].transpose(1, 2),  # (B, D_a, Sa)
                "video_features": batch["video_tokens"].transpose(1, 2),  # (B, D_v, Sv)
            }

    return IdentityBackbone()


class MainAVWrapperStub(nn.Module):
    """
    [TODO YOU] Replace this stub with your actual pre-architecture wrapper.

    In your real code this logic likely lives in `main_preprocessing_block.py`
    and does things like:
        - token alignment,
        - positional embedding,
        - modality-specific reshaping before Swin.

    Here we just pass through the batch dict unchanged.
    """

    def forward(self, batch):
        # In the real implementation you might:
        #   - read batch["audio_tokens"], batch["video_tokens"]
        #   - add positional encodings
        #   - reshape to (B, C, T, H, W) or similar for Swin
        # For now, we just return the batch as "inputs" for the backbone stub.
        return batch


# ----------------------------------
# CONFIG HELPERS
# ----------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train AV pretraining system")

    # Data
    parser.add_argument("--index_json", type=str, required=True,
                        help="Path to index JSON for AVSegmentTokensDataset")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Optional root dir to prepend to paths in index JSON")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    # Model / feature dims (YOU MUST SET THESE CORRECTLY)
    parser.add_argument("--d_a", type=int, default=256,
                        help="Audio feature dim after Swin/positional encoder")
    parser.add_argument("--d_v", type=int, default=256,
                        help="Video feature dim after Swin/positional encoder")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Sequence length S for VACL")
    parser.add_argument("--vacl_k", type=int, default=128,
                        help="Hidden size k inside VACLVA")
    parser.add_argument("--proj_out_dim", type=int, default=256,
                        help="VACL projection head output dim (fed to downstream modules)")
    parser.add_argument("--fa_common_dim", type=int, default=256,
                        help="Common space dimension d_fa for Face-Audio CPE")

    # Loss weights
    parser.add_argument("--lambda_vacl", type=float, default=1.0)
    parser.add_argument("--lambda_cpe", type=float, default=1.0)

    # Optim / training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--use_plateau_scheduler", action="store_true", default=True)

    # Logging / checkpointing
    parser.add_argument("--log_dir", type=str, default="logs_av_pretrain")
    parser.add_argument("--default_root_dir", type=str, default="checkpoints_av_pretrain")
    parser.add_argument("--experiment_name", type=str, default="av_pretrain_run")

    # Hardware
    parser.add_argument("--devices", type=str, default="auto",
                        help='Number of GPUs, "auto", or list like "0,1,2,3"')
    parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_false")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ----------------------------------
# MODEL BUILDING
# ----------------------------------


def build_architecture(args) -> AVPretrainArchitecture:
    """
    Build the full AVPretrainArchitecture from submodules.

    This function assumes:
      - AVSegmentTokenDataModule produces batches with keys:
          "audio_tokens": (B, Sa, D_a)
          "video_tokens": (B, Sv, D_v)
      - MainAVWrapper (your real wrapper) takes the raw batch and prepares
        inputs for the Swin backbone.
      - Swin backbone returns dict with:
          "audio_features": (B, D_a, S)
          "video_features": (B, D_v, S)
      - VACLProjectionHead consumes those features (layout handled internally).
      - FaceAudioCommonSpaceWrapper consumes pooled or per-frame embeddings
        from the same feature space (you can plug this into your architecture).
    """
    # [1] Build your AV wrapper (positional encoding / pre-architecture logic)
    # TODO YOU: replace MainAVWrapperStub with your real class.
    av_wrapper = MainAVWrapperStub()

    # [2] Build your Swin backbone
    # TODO YOU: replace build_swin_backbone_stub with your actual builder.
    swin_backbone = build_swin_backbone_stub(d_model=args.d_v)

    # [3] Build VACL projection head (Module A)
    module_a = VACLProjectionHead(
        d_v=args.d_v,
        d_a=args.d_a,
        seq_len=args.seq_len,
        k=args.vacl_k,
        out_dim=args.proj_out_dim,
        mu=0.5,
        input_layout="bds",  # or "bsd" depending on your Swin output; adjust as needed
        pool="mean",
    )

    # [4] Build Face-Audio common space wrapper (Module B)
    # Here we assume the common space dimension equals proj_out_dim,
    # but you can adjust as needed.
    module_b = FaceAudioCommonSpaceWrapper(
        d_a=args.d_a,
        d_f=args.d_v,
        d_fa=args.fa_common_dim,
        temperature=0.1,
        loss_weight=1.0,
    )

    # [5] Architecture-level config (optional but useful)
    arch_cfg = ArchitectureConfig(
        vacl_weight=args.lambda_vacl,
        ec_weight=args.lambda_cpe,
        enable_vacl=True,
        enable_ec=True,
        freeze_swin=False,
    )

    # [6] Build the high-level architecture
    arch = AVPretrainArchitecture(
        av_wrapper=av_wrapper,
        swin_backbone=swin_backbone,
        module_a=module_a,
        module_b=module_b,
        cfg=arch_cfg,
    )

    return arch


def build_system(args) -> AVPretrainSystem:
    """
    Wrap the architecture in AVPretrainSystem (LightningModule).
    """
    arch = build_architecture(args)

    system = AVPretrainSystem(
        architecture=arch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_vacl=args.lambda_vacl,
        lambda_cpe=args.lambda_cpe,
        use_plateau_scheduler=args.use_plateau_scheduler,
    )
    return system


# ----------------------------------
# MAIN
# ----------------------------------


def main():
    args = parse_args()

    # Reproducibility
    pl.seed_everything(args.seed, workers=True)

    # DataModule for tokenised AV segments
    datamodule = AVSegmentTokenDataModule(
        index_json_path=args.index_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root_dir=args.data_root,
        drop_last=True,
    )

    # Build model
    system = build_system(args)

    # Logger & callbacks
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.default_root_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=args.strategy if torch.cuda.is_available() else None,
        precision=args.precision,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        default_root_dir=args.default_root_dir,
        log_every_n_steps=50,
    )

    # Fit
    trainer.fit(system, datamodule=datamodule)

    # Optional: test best checkpoint
    if args.default_root_dir is not None:
        print("Training finished. You can run `trainer.test` later with the best checkpoint.")


if __name__ == "__main__":
    main()
