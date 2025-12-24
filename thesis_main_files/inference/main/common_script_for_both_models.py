# train/main/main_eval_two_models_common_trainer.py
# ============================================================
# [COMMON EVAL TRAINER] Two-model evaluation on the SAME batches
#
# Goal (Option A / "super-batch"):
#   - Use the (patched) fine-tune DataModule as the SINGLE source of truth
#   - It returns tensors for Stage-2 AND rel-paths for LAVDF boundary evaluator
#   - Run BOTH models on the SAME val_dataloader ordering, producing:
#       * Model-A (LAVDF boundary model) JSONL
#       * Model-B (Stage-2 finetune) JSONL
#
# Key design choices:
#   1) One DataModule, one seed, one split, one ordering
#   2) A tiny batch adapter adds:
#        - video_paths (absolute) computed from video_rel
#      so LAVDFEvalSystem can stay unchanged
#   3) Run sequentially in one process:
#        trainer.validate(model_A, dataloaders=val_loader)
#        trainer.validate(model_B, dataloaders=val_loader, ckpt_path=stage2_ckpt)
#
# Assumptions:
#   - You have patched scripts/dataloaders/dataloader_fine_tune.py to add:
#       batch["video_rel"], batch["audio96_rel"], batch["audio2048_rel"]
#   - LAVDFEvalSystem expects batch keys:
#       batch["video_paths"] : List[str]
#       batch["y"]           : Tensor[B]
#       optional batch["clip_ids"]
#   - Stage-2 evaluator expects:
#       batch["video_u8_cthw"] or batch["video"], batch["audio_96"] or batch["audio"], batch["y"/"label"]
#     (you already have your own evaluator; we keep it as-is)
#
# Usage example:
#   cd thesis_main_files
#   PYTHONPATH="$PWD" python train/main/main_eval_two_models_common_trainer.py \
#       --offline-root data/processed/LAV_DF/LAV_DF_training_files \
#       --batch-name lavdf_video_stage1 \
#       --segments-csv data/processed/LAV_DF/LAV_DF_training_files/lavdf_video_stage1/segment_index_finetune.csv \
#       --ckpt-a /path/to/batfd_plus_ckpt.pt \
#       --ckpt-b tb_logs/stage2_finetune/version_0/checkpoints/last.ckpt \
#       --jsonl-a outputs/modelA_lavdf.jsonl \
#       --jsonl-b outputs/modelB_stage2.jsonl \
#       --devices 1 --batch-size 2 --num-workers 4
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger


# ============================================================
# Model A: LAVDF boundary evaluator (path-based)
# ============================================================
from core.evaluation_systems.architectures.lavdf_infer_architecture import (
    LAVDFInferArchitecture,
    LAVDFInferArchitectureConfig,
)
from core.evaluation_systems.systems.lavdf_evaluation_system import (
    LAVDFEvalSystem,
    EvalSystemConfig as LavdfEvalCfg,
)

# ============================================================
# Model B: Stage-2 evaluator (tensor-based)
# Change imports ONLY if your actual file paths differ.
# ============================================================
from core.evaluation_systems.systems.fine_tune_evaluator import (
    AVFineTuneEvaluator,
    EvalConfig as Stage2EvalCfg,
)
from scripts.feature_extraction.SWIN.wrapper.main_wrapper_swins import build_backbones_for_training
from core.training_systems.architectures.finetune_architecture import (
    FinetuneArchitecture,
    FinetuneArchitectureConfig,
)

# ============================================================
# Shared DataModule: the fine-tune loader you just patched
# ============================================================
from scripts.dataloaders.dataloader_fine_tune import (
    SegmentDataModuleFineTune,
    SegmentDataModuleFineTuneConfig,
)


# ============================================================
# [ADAPTER] Add Model-A-required keys to the shared batch
#   - Adds: batch["video_paths"] computed from batch["video_rel"]
#   - Keeps everything else unchanged (super-batch)
# ============================================================
class _BatchAdapterLoader:
    def __init__(self, base_loader, *, batch_dir: Path) -> None:
        self.base_loader = base_loader
        self.batch_dir = Path(batch_dir)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in self.base_loader:
            # batch is a dict from collate_segments_pad
            if isinstance(batch, dict):
                # Provide "video_paths" for the LAVDF system
                if "video_paths" not in batch:
                    v_rel = batch.get("video_rel", None)
                    if v_rel is not None:
                        # v_rel is List[str] (RELATIVE to <offline_root>/<batch_name>)
                        batch["video_paths"] = [str((self.batch_dir / p).resolve()) for p in v_rel]
            yield batch

    def __len__(self) -> int:
        # Lightning may use __len__ for progress bar / sanity checks
        return len(self.base_loader)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # -------------------------
    # Data (shared)
    # -------------------------
    ap.add_argument("--offline-root", required=True, type=str)
    ap.add_argument("--batch-name", required=True, type=str)
    ap.add_argument("--segments-csv", default=None, type=str)  # segment_index_finetune.csv (with label column)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)

    # -------------------------
    # Model A (LAVDF / BATFD)
    # -------------------------
    ap.add_argument("--ckpt-a", required=True, type=str)
    ap.add_argument("--model-a-type", default="batfd_plus", choices=["batfd", "batfd_plus"])
    ap.add_argument("--model-a-strict-load", action="store_true")
    ap.add_argument("--thr-a", type=float, default=0.5)
    ap.add_argument("--jsonl-a", required=True, type=str)

    # -------------------------
    # Model B (Stage-2 finetune)
    # -------------------------
    ap.add_argument("--ckpt-b", required=True, type=str)  # Lightning .ckpt
    ap.add_argument("--thr-b", type=float, default=0.5)
    ap.add_argument("--jsonl-b", required=True, type=str)

    # -------------------------
    # Trainer
    # -------------------------
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", default="bf16-mixed")

    # -------------------------
    # TensorBoard
    # -------------------------
    ap.add_argument("--tb-save-dir", default="outputs/tb")
    ap.add_argument("--tb-name", default="common_two_model_eval")
    ap.add_argument("--tb-version", default=None)

    return ap


def main() -> None:
    args = build_argparser().parse_args()
    pl.seed_everything(int(args.seed), workers=True)

    offline_root = Path(args.offline_root)
    batch_name = str(args.batch_name)
    batch_dir = offline_root / batch_name

    # ============================================================
    # 1) Shared DataModule (fine-tune "super-batch")
    # ============================================================
    dm_cfg = SegmentDataModuleFineTuneConfig(
        offline_root=offline_root,
        batch_name=batch_name,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        val_split=float(args.val_split),
        seed=int(args.seed),
        segments_csv=Path(args.segments_csv) if args.segments_csv else None,
        strict_labels=True,
    )
    dm = SegmentDataModuleFineTune(dm_cfg)
    dm.setup()

    # Use VAL loader for both models (same order)
    base_val_loader = dm.val_dataloader()

    # Adapter adds "video_paths" derived from "video_rel" (keeps batch unchanged otherwise)
    val_loader = _BatchAdapterLoader(base_val_loader, batch_dir=batch_dir)

    # ============================================================
    # 2) Build Model A (LAVDF boundary)
    # ============================================================
    arch_a = LAVDFInferArchitecture(
        cfg=LAVDFInferArchitectureConfig(model_type=args.model_a_type, return_prob=True),
        ckpt_path=args.ckpt_a,
        strict_load=bool(args.model_a_strict_load),
    )
    sys_a = LAVDFEvalSystem(
        arch=arch_a,
        cfg=LavdfEvalCfg(
            threshold=float(args.thr_a),
            save_preds_path=None,
            save_jsonl_path=str(args.jsonl_a),
        ),
    )

    # ============================================================
    # 3) Build Model B (Stage-2 architecture + evaluator wrapper)
    # NOTE: weights loaded via ckpt_path in trainer.validate(...)
    # ============================================================
    video_backbone, audio_backbone = build_backbones_for_training()

    arch_b = FinetuneArchitecture(
        video_backbone=video_backbone,
        audio_backbone=audio_backbone,
        cfg=FinetuneArchitectureConfig(),
        c_v_in=256,
        c_a_in=768,
    )

    # IMPORTANT:
    # - Use your evaluator’s JSONL option (you named it save_preds_path earlier)
    # - Ensure mlp_hidden is NEVER -1 (use None for linear head)
    sys_b = AVFineTuneEvaluator(
        model=arch_b,
        cfg=Stage2EvalCfg(
            pool="mean",
            use_layernorm=False,
            mlp_hidden=None,
            dropout=0.0,
            compute_loss=True,
            save_preds_path=str(args.jsonl_b),
        ),
    )

    # ============================================================
    # 4) Trainer (single instance)
    # ============================================================
    tb_logger = TensorBoardLogger(
        save_dir=args.tb_save_dir,
        name=args.tb_name,
        version=args.tb_version,
    )

    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator=args.accelerator,
        devices=int(args.devices),
        precision=str(args.precision),
        enable_checkpointing=False,
    )

    # ============================================================
    # 5) Evaluate both models on SAME val_loader
    # ============================================================
    print("[COMMON EVAL] Running Model A (LAVDF boundary) ...")
    trainer.validate(sys_a, dataloaders=val_loader, verbose=True)

    print("[COMMON EVAL] Running Model B (Stage-2 finetune) ...")
    trainer.validate(sys_b, dataloaders=val_loader, ckpt_path=str(args.ckpt_b), verbose=True)

    print("[DONE]")
    print(f"  JSONL A: {Path(args.jsonl_a).resolve()}")
    print(f"  JSONL B: {Path(args.jsonl_b).resolve()}")
    print(f"[TB] tensorboard --logdir {Path(args.tb_save_dir).resolve()}")


if __name__ == "__main__":
    # Helpful for CUDA determinism (optional)
    torch.set_float32_matmul_precision("high")
    main()
