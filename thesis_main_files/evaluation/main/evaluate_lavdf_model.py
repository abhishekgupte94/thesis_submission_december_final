# train/main/main_stage_lavdf_eval.py
# ============================================================
# [FINAL PATCHED MAIN] Stage-LAVDF Evaluation Entry Point
#
# Wires:
#   - Existing DataModule (unchanged)
#   - LAVDFInferArchitecture
#   - LAVDFEvalSystem (TensorBoard + CSV + JSONL)
#   - Lightning Trainer
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from core.evaluation_systems.architectures.lavdf_infer_architecture import (
    LAVDFInferArchitecture,
    LAVDFInferArchitectureConfig,
)
from core.evaluation_systems.systems.lavdf_evaluation_system import (
    LAVDFEvalSystem,
    EvalSystemConfig,
)

# >>> CHANGE THIS to your real DataModule import if different <<<
from scripts.dataloaders.dataloader_boundary_module import (
    AVPathsDataModule,
    AVPathsDataModuleConfig,
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-type", default="batfd_plus", choices=["batfd", "batfd_plus"])
    ap.add_argument("--strict-load", action="store_true")

    # Evaluation
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save-preds", type=str, default=None)
    ap.add_argument("--save-jsonl", type=str, default=None)

    # TensorBoard
    ap.add_argument("--tb-save-dir", default="outputs/tb")
    ap.add_argument("--tb-name", default="lavdf_eval")
    ap.add_argument("--tb-version", default=None)

    # Trainer
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", default="bf16-mixed")

    # DataModule
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--batch-name", required=True)
    ap.add_argument("--index-csv", default=None)
    ap.add_argument("--labels-csv", default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)

    return ap


def main() -> None:
    args = build_argparser().parse_args()
    pl.seed_everything(int(args.seed), workers=True)

    # DataModule (unchanged)
    dm_cfg = AVPathsDataModuleConfig(
        offline_root=Path(args.offline_root),
        batch_name=args.batch_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        index_csv=Path(args.index_csv) if args.index_csv else None,
        labels_csv=Path(args.labels_csv) if args.labels_csv else None,
        val_split=args.val_split,
        seed=args.seed,
    )
    dm = AVPathsDataModule(dm_cfg)
    dm.setup()

    # Architecture
    arch_cfg = LAVDFInferArchitectureConfig(
        model_type=args.model_type,
        return_prob=True,
    )
    arch = LAVDFInferArchitecture(
        cfg=arch_cfg,
        ckpt_path=args.ckpt,
        strict_load=args.strict_load,
    )

    # System
    sys_cfg = EvalSystemConfig(
        threshold=args.threshold,
        save_preds_path=args.save_preds,
        save_jsonl_path=args.save_jsonl,
    )
    system = LAVDFEvalSystem(arch=arch, cfg=sys_cfg)

    # Logger + Trainer
    tb_logger = TensorBoardLogger(
        save_dir=args.tb_save_dir,
        name=args.tb_name,
        version=args.tb_version,
    )

    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        enable_checkpointing=False,
    )

    trainer.validate(system, dataloaders=dm.val_dataloader(), verbose=True)
    print(f"[TB] tensorboard --logdir {Path(args.tb_save_dir).resolve()}")


if __name__ == "__main__":
    main()
