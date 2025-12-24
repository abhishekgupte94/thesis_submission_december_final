# thesis_main_files/train/eval_stage2.py
# ============================================================
# [DROP-IN] MAIN EVAL SCRIPT - STAGE 2
#
# GOAL:
#   Mirror train_stage2_finetune.py wiring, but for evaluation:
#     1) Resolve repo-relative paths
#     2) Create Stage-2 DataModule (same as finetune)
#     3) Build backbones (same helper as Stage-1/Stage-2 trainer)
#     4) Build Stage-2 architecture (same as finetune)
#     5) Wrap in AVFineTuneEvaluator (eval-only LightningModule)
#     6) Run validate/test/predict using ckpt_path
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from scripts.dataloaders.dataloader_fine_tune import (
    SegmentDataModuleFineTune,
    SegmentDataModuleFineTuneConfig,
)
from scripts.feature_extraction.SWIN.wrapper.main_wrapper_swins import build_backbones_for_training

from core.training_systems.architectures.finetune_architecture import (
    FinetuneArchitecture,
    FinetuneArchitectureConfig,
)

from evaluation_systems.systems.fine_tune_evaluator import (
    AVFineTuneEvaluator,
    EvalConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]  # match Stage-2 trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # ============================================================
    # [MIRRORED] Path resolution like finetune trainer
    # ============================================================
    p.add_argument("--evaluation-root", type=str, required=True)
    p.add_argument("--batch-name", type=str, required=True)

    # checkpoint
    p.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt produced by Stage-2 trainer")

    # eval mode
    p.add_argument("--mode", type=str, default="validate", choices=["validate", "test", "predict"])

    # dataloader knobs (mirror)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-split", type=float, default=0.05)

    # precision
    p.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
    )

    # tensorboard logging
    p.add_argument("--tb-logdir", type=str, default="tb_logs")
    p.add_argument("--run-name", type=str, default="stage2_eval")

    # evaluator head knobs (mirror of stage2 head args)
    p.add_argument("--stage2-pool", type=str, default="mean", choices=["mean", "max"])
    p.add_argument("--stage2-use-layernorm", action="store_true")
    p.add_argument("--stage2-mlp-hidden", type=int, default=-1)  # -1 => None
    p.add_argument("--stage2-dropout", type=float, default=0.0)

    # optional prediction dump (predict mode)
    p.add_argument("--save-preds-path", type=str, default="")

    return p.parse_args()


def _resolve_repo_relative(p: str) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((REPO_ROOT / pp).resolve())


def main() -> None:
    args = parse_args()

    # matmul precision hint (mirrors your trainer)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    evaluation_root = _resolve_repo_relative(args.evaluation_root)

    # ============================================================
    # DataModule (same class as Stage-2 trainer)
    # NOTE: this assumes your eval_root mirrors the offline_root structure:
    #   <evaluation_root>/<batch_name>/{audio, video_face_crops, segment_paths_finetune.csv, ...}
    # ============================================================
    dm_cfg = SegmentDataModuleFineTuneConfig(
        offline_root=Path(evaluation_root),  # <- keep cfg field name; you're using eval root
        batch_name=args.batch_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        # if your config supports them, they should remain enabled for speed:
        # pin_memory=True,
        # persistent_workers=(args.num_workers > 0),
    )
    dm = SegmentDataModuleFineTune(cfg=dm_cfg)

    # ============================================================
    # Build backbones + Stage-2 architecture exactly like finetune trainer
    # ============================================================
    video_backbone, audio_backbone = build_backbones_for_training()

    ft_cfg = FinetuneArchitectureConfig(
        vacl_s_out=64,
        vacl_d_v=768,
        vacl_d_a=768,
        compute_infonce=True,
        return_intermediates=False,
        cpe_d_common=512,
    )

    model = FinetuneArchitecture(
        video_backbone=video_backbone,
        audio_backbone=audio_backbone,
        cfg=ft_cfg,
        c_v_in=768,
        c_a_in=768,
    )

    # ============================================================
    # Evaluator (Stage-2 head built lazily on first batch)
    # ============================================================
    eval_cfg = EvalConfig(
        pool=args.stage2_pool,
        use_layernorm=bool(args.stage2_use_layernorm),
        mlp_hidden=None if int(args.stage2_mlp_hidden) < 0 else int(args.stage2_mlp_hidden),
        dropout=float(args.stage2_dropout),
        compute_loss=True,
        save_preds_path=(args.save_preds_path or None),
    )

    system = AVFineTuneEvaluator(model=model, cfg=eval_cfg)

    # ============================================================
    # Logger + Trainer
    # ============================================================
    logger = TensorBoardLogger(
        save_dir=_resolve_repo_relative(args.tb_logdir),
        name=args.run_name,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        precision=args.precision,
        logger=logger,
        enable_checkpointing=False,
        log_every_n_steps=10,
    )

    # ============================================================
    # Run (Lightning restores weights from ckpt_path automatically)
    # ============================================================
    if args.mode == "validate":
        trainer.validate(model=system, datamodule=dm, ckpt_path=str(ckpt_path))
    elif args.mode == "test":
        trainer.test(model=system, datamodule=dm, ckpt_path=str(ckpt_path))
    else:
        trainer.predict(model=system, datamodule=dm, ckpt_path=str(ckpt_path))


if __name__ == "__main__":
    main()
