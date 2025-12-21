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
#
# ============================================================
# ===================== [GRID SEARCH] Additions =====================
# Description:
#   Optional grid-search mode that runs multiple sequential trials:
#     - Each trial builds the *same* datamodule/model/system/trainer flow
#     - DDP strategy remains intact per trial
#     - Defaults preserve original single-run behavior
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
from core.training_systems.training_systems.system_fine import AVPretrainSystem

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


# ============================================================
# ===================== [GRID SEARCH] Helpers =====================
# Description:
#   Tiny parsing helpers for comma-separated CLI grids.
#   No external deps.
# ============================================================
def _parse_csv_list(s: str, cast_fn):
    items = [x.strip() for x in str(s).split(",") if x.strip() != ""]
    return [cast_fn(x) for x in items]


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

    # ============================================================
    # ===================== [GRID SEARCH] Stage-2 Head knobs =====================
    # Description:
    #   Single-run overrides (also used by each grid trial).
    #   Defaults preserve your current head behavior.
    # ============================================================
    p.add_argument("--stage2-pool", type=str, default="mean", choices=["mean", "max"])
    p.add_argument("--stage2-use-layernorm", action="store_true")
    p.add_argument("--stage2-mlp-hidden", type=int, default=-1)  # -1 => None
    p.add_argument("--stage2-dropout", type=float, default=0.0)

    # ============================================================
    # ===================== [GRID SEARCH] Optimizer param-group knobs =====================
    # Description:
    #   Optional separate LR/WD for head vs other trainables.
    # ============================================================
    p.add_argument("--lr-head", type=float, default=None)
    p.add_argument("--weight-decay-head", type=float, default=None)
    p.add_argument("--lr-backbone", type=float, default=None)
    p.add_argument("--weight-decay-backbone", type=float, default=None)

    # ============================================================
    # ===================== [GRID SEARCH] Mode toggles + speed controls =====================
    # Description:
    #   Stage-1 grid search usually runs:
    #     - fewer GPUs (often 1)
    #     - fewer epochs (often 1)
    #     - limit_train/val_batches < 1.0
    #   These do not change math; they only reduce compute for screening.
    # ============================================================
    p.add_argument("--grid-search", action="store_true")
    p.add_argument("--grid-devices", type=int, default=1)
    p.add_argument("--grid-max-epochs", type=int, default=1)
    p.add_argument("--grid-limit-train-batches", type=float, default=1.0)
    p.add_argument("--grid-limit-val-batches", type=float, default=1.0)

    # ============================================================
    # ===================== [GRID SEARCH] Search spaces (comma-separated) =====================
    # Description:
    #   If empty, each grid dimension becomes a singleton using the base arg value.
    #   This preserves original single-run behavior when --grid-search is OFF.
    # ============================================================
    p.add_argument("--grid-lr", type=str, default="")
    p.add_argument("--grid-weight-decay", type=str, default="")
    p.add_argument("--grid-omega", type=str, default="")
    p.add_argument("--grid-lambda_", type=str, default="")
    p.add_argument("--grid-beta", type=str, default="")

    p.add_argument("--grid-stage2-pool", type=str, default="")
    p.add_argument("--grid-stage2-use-layernorm", type=str, default="")
    p.add_argument("--grid-stage2-mlp-hidden", type=str, default="")
    p.add_argument("--grid-stage2-dropout", type=str, default="")

    p.add_argument("--grid-lr-head", type=str, default="")
    p.add_argument("--grid-weight-decay-head", type=str, default="")
    p.add_argument("--grid-lr-backbone", type=str, default="")
    p.add_argument("--grid-weight-decay-backbone", type=str, default="")

    return p.parse_args()
