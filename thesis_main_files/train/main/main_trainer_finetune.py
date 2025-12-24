# thesis_main_files/train/train_stage2_finetune.py
# ============================================================
# [FINAL PATCHED] MAIN TRAIN SCRIPT - STAGE 2 FINETUNE
#
# GOAL OF THIS PATCH:
#   - Mirror the Stage-1 (pretrain) Lightning wiring in this Stage-2 finetuner:
#       1) Resolve repo-relative paths (prevents saving under wrong cwd)
#       2) Create DataModule (train/val split + any internal indexing)
#       3) Build backbones (reuse the same helper as Stage-1)
#       4) Build finetune architecture (Stage-2 model)
#       5) Build Lightning System (system_finetune.AVFineTuneSystem)
#       6) Build Trainer (DDP + bf16-mixed + TensorBoard + checkpoints)
#
#   - KEEP Stage-2 specific knobs (stage2-* + grid-* args) intact.
#   - Provide a single, explicit place to plug your Stage-2 architecture builder
#     without changing the Lightning flow.
# ============================================================

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# NOTE: keep your existing imports below as-is in your repo:
from scripts.dataloaders.dataloader_fine_tune import SegmentDataModuleFineTune,SegmentDataModuleFineTuneConfig # [MIRRORED] Stage-2 dataloader
from core.training_systems.training_systems.system_finetune import AVFineTuneSystem  # [MIRRORED] Stage-2 system

# [MIRRORED][FROM STAGE-1] Backbone builder helper (keeps init identical)
from scripts.feature_extraction.SWIN.wrapper.main_wrapper_swins import build_backbones_for_training

# ============================================================
# [STAGE-2] Your finetune module(s) / blocks can be imported here.
# IMPORTANT:
#   - Do NOT change Lightning wiring below.
#   - Only change the Stage-2 architecture builder function if needed.
# ============================================================

# Example: if Stage-2 uses your fine-tune VACL wrapper directly
from core.NPVForensics.VACL_block.main.vacl_wrapper_fine_tune import (
    VACLWrapper,
    VACLWrapperConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]  # match Stage-1


# ============================================================
# ===================== [GRID SEARCH] Helpers =====================
# ============================================================
def _parse_csv_list(s: str, *, cast: Any = float) -> List[Any]:
    """Parse a comma-separated list. Empty string => empty list."""
    s = (s or "").strip()
    if not s:
        return []
    out: List[Any] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(cast(tok))
    return out


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

    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-2)

    # ============================================================
    # [ADDED][STAGE-2] Loss weights (consumed by system_finetune.AVFineTuneSystem)
    # ============================================================
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--lambda_", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=1.0)

    # ============================================================
    # [ADDED][STAGE-2] Validation memory guards
    # ============================================================
    p.add_argument("--val-auc-thresholds", type=tuple, default= (0.5,))
    p.add_argument("--val-metric-cap-batches", type=int, default=200)

    # ============================================================
    # [MIRRORED][FROM STAGE-1] Runtime knobs (no wiring changes)
    # ============================================================
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--mem-log-every", type=int, default=50)
    p.add_argument("--smoke-test", action="store_true")

    # precision (A100: bf16-mixed recommended)
    p.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
    )

    # validation split
    p.add_argument("--val-split", type=float, default=0.05)

    # profiling toggles
    p.add_argument("--enable-energy-tracking", action="store_true")
    p.add_argument("--enable-flops-profile", action="store_true")

    # tensorboard logging
    p.add_argument("--tb-logdir", type=str, default="tb_logs")
    p.add_argument("--run-name", type=str, default="stage2_finetune")

    # ============================================================
    # ===================== [STAGE-2] Single-run overrides =====================
    # Defaults preserve your current head behavior.
    # ============================================================
    p.add_argument("--stage2-pool", type=str, default="mean", choices=["mean", "max"])
    p.add_argument("--stage2-use-layernorm", action="store_true")
    p.add_argument("--stage2-mlp-hidden", type=int, default=-1)  # -1 => None
    p.add_argument("--stage2-dropout", type=float, default=0.0)

    # ============================================================
    # ===================== [GRID SEARCH] Loss-weight knobs =====================
    # Provide comma-separated lists. Empty => no grid over that knob.
    # ============================================================
    p.add_argument("--grid-omega", type=str, default="")
    p.add_argument("--grid-lambda_", type=str, default="")
    p.add_argument("--grid-beta", type=str, default="")

    # ============================================================
    # ===================== [GRID SEARCH] Stage-2 head knobs =====================
    # ============================================================
    p.add_argument("--grid-stage2-pool", type=str, default="")
    p.add_argument("--grid-stage2-use-layernorm", type=str, default="")
    p.add_argument("--grid-stage2-mlp-hidden", type=str, default="")
    p.add_argument("--grid-stage2-dropout", type=str, default="")

    # ============================================================
    # ===================== [GRID SEARCH] Optimizer param-group knobs =====================
    # Optional separate LR/WD for head vs other trainables.
    # ============================================================
    p.add_argument("--lr-head", type=float, default=None)
    p.add_argument("--weight-decay-head", type=float, default=None)
    p.add_argument("--lr-backbone", type=float, default=None)
    p.add_argument("--weight-decay-backbone", type=float, default=None)

    p.add_argument("--grid-lr-head", type=str, default="")
    p.add_argument("--grid-weight-decay-head", type=str, default="")
    p.add_argument("--grid-lr-backbone", type=str, default="")
    p.add_argument("--grid-weight-decay-backbone", type=str, default="")

    # [ADDED] Optional resume checkpoint
    p.add_argument(
        "--ckpt-path",
        type=str,
        default="",
        help="Optional: path to a Lightning .ckpt to resume training from.",
    )

    return p.parse_args()


# ============================================================
# [STAGE-2][DROP-IN] Build Stage-2 architecture
#   - This is the ONLY function you should edit when you wire your new nodule.
#   - Keep signature + return contract stable.
# ============================================================
def _iter_grid_trials(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    """Yield dicts of parameter overrides for grid search. Empty grid => one empty dict."""

    omega_list = _parse_csv_list(args.grid_omega, cast=float) or [None]
    lambda_list = _parse_csv_list(args.grid_lambda_, cast=float) or [None]
    beta_list = _parse_csv_list(args.grid_beta, cast=float) or [None]

    pool_list = [s.strip() for s in (args.grid_stage2_pool or "").split(",") if s.strip()] or [None]
    ln_list = _parse_csv_list(args.grid_stage2_use_layernorm, cast=int) or [None]
    mlp_list = _parse_csv_list(args.grid_stage2_mlp_hidden, cast=int) or [None]
    drop_list = _parse_csv_list(args.grid_stage2_dropout, cast=float) or [None]

    lrh_list = _parse_csv_list(args.grid_lr_head, cast=float) or [None]
    wdh_list = _parse_csv_list(args.grid_weight_decay_head, cast=float) or [None]
    lrb_list = _parse_csv_list(args.grid_lr_backbone, cast=float) or [None]
    wdb_list = _parse_csv_list(args.grid_weight_decay_backbone, cast=float) or [None]

    any_grid = any(
        x not in ("", None)
        for x in [
            args.grid_omega,
            args.grid_lambda_,
            args.grid_beta,
            args.grid_stage2_pool,
            args.grid_stage2_use_layernorm,
            args.grid_stage2_mlp_hidden,
            args.grid_stage2_dropout,
            args.grid_lr_head,
            args.grid_weight_decay_head,
            args.grid_lr_backbone,
            args.grid_weight_decay_backbone,
        ]
    )

    if not any_grid:
        yield {}
        return

    for (
        omega,
        lambda_,
        beta,
        pool,
        use_ln_i,
        mlp_hidden,
        dropout,
        lr_head,
        wd_head,
        lr_backbone,
        wd_backbone,
    ) in itertools.product(
        omega_list,
        lambda_list,
        beta_list,
        pool_list,
        ln_list,
        mlp_list,
        drop_list,
        lrh_list,
        wdh_list,
        lrb_list,
        wdb_list,
    ):
        yield {
            "omega": omega,
            "lambda_": lambda_,
            "beta": beta,
            "stage2_pool": pool,
            "stage2_use_layernorm": None if use_ln_i is None else bool(int(use_ln_i)),
            "stage2_mlp_hidden": mlp_hidden,
            "stage2_dropout": dropout,
            "lr_head": lr_head,
            "weight_decay_head": wd_head,
            "lr_backbone": lr_backbone,
            "weight_decay_backbone": wd_backbone,
        }


def main() -> None:
    args = parse_args()

    # [MIRRORED][FROM STAGE-1] matmul precision hint
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
    # DataModule (MIRRORED)
    # ============================================================
    cfg_dl = SegmentDataModuleFineTuneConfig(offline_root=offline_root,
        batch_name=args.batch_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # pin_memory=True,
        # persistent_workers=True,
        val_split=args.val_split,)
    dm = SegmentDataModuleFineTune(
            cfg= cfg_dl
    )

    # ============================================================
    # TensorBoard logger (MIRRORED)
    # ============================================================
    base_logger = TensorBoardLogger(
        save_dir=str((REPO_ROOT / args.tb_logdir).resolve()),
        name=args.run_name,
    )

    # ============================================================
    # Trainer config (MIRRORED)
    # ============================================================
    def _run_one_trial(overrides: Dict[str, Any]) -> None:
        # ---- merge overrides into a shallow, local view (no mutation of args) ----
        local = argparse.Namespace(**vars(args))
        for k, v in overrides.items():
            if v is None:
                continue
            setattr(local, k, v)

        # ---- build backbones exactly like Stage-1 ----
        video_backbone, audio_backbone = build_backbones_for_training()

        # ---- build Stage-2 architecture (YOU wire this function) ----
        # ============================================================
# ============================================================
        # [STAGE-2 MODEL] Instantiate your Stage-2 finetune architecture HERE
        # IMPORTANT: This mirrors Stage-1: build pure nn.Module, then wrap with Lightning system.
        from core.training_systems.architectures.finetune_architecture import (
            FinetuneArchitecture,
            FinetuneArchitectureConfig,
        )

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
            c_v_in=256,
            c_a_in=768,
        )


        # ---- Lightning System (mirror Stage-1 wiring) ----
        system = AVFineTuneSystem(
            model=model,
            lr=local.lr,
            weight_decay=local.weight_decay,
            # ------------------------------------------------------------
            # [STAGE-2] Loss weights
            # ------------------------------------------------------------
            omega=getattr(local, 'omega', 1.0),
            lambda_cpe=getattr(local, 'lambda_', 1.0),
            alpha=getattr(local, 'alpha', 0.0),
            beta=getattr(local, 'beta', 1.0),
            # ------------------------------------------------------------
            # [STAGE-2] Validation guards
            # ------------------------------------------------------------
            val_auc_thresholds=getattr(local, 'val_auc_thresholds', 256),
            val_metric_cap_batches=getattr(local, 'val_metric_cap_batches', 200),
            # ------------------------------------------------------------
            # [GRID SEARCH] Head hyperparams
            # ------------------------------------------------------------
            stage2_pool=getattr(local, 'stage2_pool', 'mean'),
            stage2_use_layernorm=bool(getattr(local, 'stage2_use_layernorm', False)),
            stage2_mlp_hidden=int(getattr(local, 'stage2_mlp_hidden', 256)),
            stage2_dropout=float(getattr(local, 'stage2_dropout', 0.0)),
            # ------------------------------------------------------------
            # [GRID SEARCH] Optimizer param-groups
            # ------------------------------------------------------------
            lr_head=getattr(local, 'lr_head', None),
            weight_decay_head=getattr(local, 'weight_decay_head', None),
            lr_backbone=getattr(local, 'lr_backbone', None),
            weight_decay_backbone=getattr(local, 'weight_decay_backbone', None),
            enable_energy_tracking=bool(local.enable_energy_tracking),
            enable_flops_profile=bool(local.enable_flops_profile),
        )

        # [MIRRORED] runtime knobs

        system.mem_log_every = int(local.mem_log_every)
        system.smoke_test = bool(local.smoke_test)

        # [MIRRORED] smoke-test overrides
        if local.smoke_test:
            devices = 1
            strategy = "auto"
            max_epochs = 1
            limit_train_batches = 2
            limit_val_batches = 1
            num_sanity_val_steps = 0
            enable_checkpointing = False
            log_every_n_steps = 1
            system.mem_log_every = 1
        else:
            devices = local.devices
            strategy = "ddp"
            max_epochs = local.max_epochs
            limit_train_batches = 1.0
            limit_val_batches = 1.0
            num_sanity_val_steps = 2
            enable_checkpointing = True
            log_every_n_steps = 10

        # ---- per-trial logger name suffix (safe; no folder explosion unless grid used) ----
        logger = base_logger
        if overrides:
            suffix_parts = []
            for k in sorted(overrides.keys()):
                v = overrides[k]
                if v is None:
                    continue
                suffix_parts.append(f"{k}={v}")
            trial_name = "grid__" + "__".join(suffix_parts) if suffix_parts else "grid"
            logger = TensorBoardLogger(
                save_dir=str((REPO_ROOT / local.tb_logdir).resolve()),
                name=f"{local.run_name}/{trial_name}",
            )

        ckpt_cb = ModelCheckpoint(
            save_last=True,
            save_top_k=-1,
            every_n_epochs=1,
            filename="epoch={epoch}-step={step}",
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=devices,
            strategy=strategy,
            precision=local.precision,
            max_epochs=max_epochs,
            logger=logger,
            check_val_every_n_epoch=1,
            callbacks=[ckpt_cb],
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            accumulate_grad_batches=local.accumulate_grad_batches,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            num_sanity_val_steps=num_sanity_val_steps,
        )

        # [ADDED] Lightning-style resume
        ckpt_path = args.ckpt_path or None

        print(f"[main_trainer_finetune] ckpt_path = {ckpt_path or 'NONE'}")

        trainer.fit(system, datamodule=dm, ckpt_path=ckpt_path)

    # ============================================================
    # Grid-search loop (optional)
    # ============================================================
    for trial_overrides in _iter_grid_trials(args):
        _run_one_trial(trial_overrides)


if __name__ == "__main__":
    main()
