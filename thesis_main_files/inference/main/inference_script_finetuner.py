# train/infer_stage2.py
from __future__ import annotations

import argparse
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.loggers import CSVLogger  # lightweight

from evaluation_systems.systems.fine_tune_evaluator import AVFineTuneEvaluator, EvalConfig

# Use your real imports here (same ones you use in finetune training):
from scripts.dataloaders.dataloader_fine_tune import SegmentDataModuleFineTune, SegmentDataModuleFineTuneConfig
from scripts.feature_extraction.SWIN.wrapper.main_wrapper_swins import build_backbones_for_training
from core.training_systems.architectures.finetune_architecture import FinetuneArchitecture, FinetuneArchitectureConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--evaluation-root", type=str, required=True)
    ap.add_argument("--batch-name", type=str, required=True)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--precision", type=str, default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    ap.add_argument("--save-preds-path", type=str, default="outputs/preds_stage2.jsonl")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")

    # Data (must provide "video" and "audio" in batch; label optional)
    dm = SegmentDataModuleFineTune(
        cfg=SegmentDataModuleFineTuneConfig(
            offline_root=args.evaluation_root,
            batch_name=args.batch_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

    # Build model exactly like training
    video_backbone, audio_backbone = build_backbones_for_training()

    arch = FinetuneArchitecture(
        video_backbone=video_backbone,
        audio_backbone=audio_backbone,
        cfg=FinetuneArchitectureConfig(),
        c_v_in=256,
        c_a_in=768,
    )

    # Evaluator wrapper (predict writes JSONL if save_preds_path is set)
    system = AVFineTuneEvaluator(
        model=arch,
        cfg=EvalConfig(
            pool="mean",
            use_layernorm=False,
            mlp_hidden=None,        # IMPORTANT: do NOT use -1 here
            dropout=0.0,
            compute_loss=False,     # inference
            save_preds_path=args.save_preds_path,
        ),
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        precision=args.precision,
        logger=CSVLogger("tb_logs", name="stage2_infer"),
        enable_checkpointing=False,
    )

    # This is where the checkpoint is actually loaded:
    trainer.predict(model=system, datamodule=dm, ckpt_path=str(ckpt))


if __name__ == "__main__":
    main()
