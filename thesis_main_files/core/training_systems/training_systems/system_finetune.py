# core/training_systems/training_systems/system_fine.py
# ============================================================
# [FINAL PATCHED] AVPretrainSystem (LightningModule)  <-- kept class name for compatibility
#
# Fresh Stage-2 design (no stage flags):
#   - Expects model.forward(...) returns dict with:
#       "X_v_att": (B,k,S)
#       "X_a_att": (B,k,S)
#       "L_cor":   scalar
#       "l_infonce": scalar   (REQUIRED)
#
# Trainer-owned:
#   - Stage2AVClassifierHead => logits + probabilities
#   - Paper Eq.18 BCE on P (ACTIVE)
#   - DDP-safe metrics: ACC / AUC / F1 (epoch aggregated)
#   - Val metric cap (first N val batches only)
#
# Loss:
#   loss_total = omega*L_cor + lambda*l_infonce + beta*L_bce (+ alpha*L_bmn if enabled)
#
# Freeze policy:
#   - Swin2D + Swin3D are frozen here (requires_grad False)
#   - Swins forced to eval() every epoch (Lightning calls train() globally otherwise)
# ============================================================

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import torch
import pytorch_lightning as pl

# ============================================================
# [ADDED] Stage-2 classifier head + paper BCE
# ============================================================
from core.training_systems.architectures.final_classifier_module import (
    Stage2AVClassifierHead,
    Stage2HeadConfig,
    bce_paper_eq18,
)

# ============================================================
# [ADDED] Metrics (DDP-safe)
# Notes:
#   - AUROC(thresholds=...) is memory-bounded
# ============================================================
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score


class AVPretrainSystem(pl.LightningModule):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        # ------------------------------------------------------------
        # [KEPT] Legacy weights (no longer used for stage-2 loss composition)
        # ------------------------------------------------------------
        lambda_vacl: float = 1.0,
        lambda_cpe: float = 1.0,
        # ============================================================
        # [ADDED] Stage-2 loss weights
        # ============================================================
        omega: float = 1.0,
        lambda_: float = 1.0,
        alpha: float = 0.0,   # L_bmn weight (inactive by default)
        beta: float = 1.0,    # BCE weight
        # ============================================================
        # [ADDED] Validation memory guards
        # ============================================================
        val_auc_thresholds: int = 256,
        val_metric_cap_batches: int = 200,

        # ============================================================
        # ===================== [GRID SEARCH] Stage-2 Head HParams =====================
        # Description:
        #   These args allow the *trainer script* to grid-search head configurations
        #   without changing the model math/wiring elsewhere.
        #   Defaults exactly preserve current behavior.
        # ==============================================================================
        stage2_pool: str = "mean",
        stage2_use_layernorm: bool = False,
        stage2_mlp_hidden: int = -1,     # -1 => None
        stage2_dropout: float = 0.0,

        # ============================================================
        # ===================== [GRID SEARCH] Optimizer Param Groups =====================
        # Description:
        #   Adds "trailing groups" parameter-group support for grid search:
        #     - head LR/WD can differ from the rest of trainable params
        #   Defaults preserve current behavior (single lr/wd everywhere).
        #   Frozen Swins remain excluded because requires_grad=False.
        # ==============================================================================
        lr_head: float | None = None,
        weight_decay_head: float | None = None,
        lr_backbone: float | None = None,
        weight_decay_backbone: float | None = None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------
        # [KEPT] Core config
        # ------------------------------------------------------------
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        # [KEPT] legacy (not used in stage2 loss equation)
        self.lambda_vacl = float(lambda_vacl)
        self.lambda_cpe = float(lambda_cpe)

        # ------------------------------------------------------------
        # [ADDED] Stage-2 weights
        # ------------------------------------------------------------
        self.omega = float(omega)
        self.lambda_ = float(lambda_)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # ------------------------------------------------------------
        # [ADDED] Validation memory guards
        # ------------------------------------------------------------
        self.val_auc_thresholds = int(val_auc_thresholds)
        self.val_metric_cap_batches = int(val_metric_cap_batches)

        # ============================================================
        # ===================== [GRID SEARCH] Save Head HParams =====================
        # Description:
        #   Stored so head construction is deterministic per run/trial.
        # ==============================================================================
        self._stage2_pool = str(stage2_pool)
        self._stage2_use_layernorm = bool(stage2_use_layernorm)
        self._stage2_mlp_hidden = int(stage2_mlp_hidden)
        self._stage2_dropout = float(stage2_dropout)

        # ============================================================
        # ===================== [GRID SEARCH] Save Optimizer Group HParams =====================
        # Description:
        #   Stored and used in configure_optimizers() param-groups.
        # ==============================================================================
        self.lr_head = None if lr_head is None else float(lr_head)
        self.weight_decay_head = None if weight_decay_head is None else float(weight_decay_head)
        self.lr_backbone = None if lr_backbone is None else float(lr_backbone)
        self.weight_decay_backbone = None if weight_decay_backbone is None else float(weight_decay_backbone)

        # ============================================================
        # [ADDED] Freeze Swin backbones (params)
        # Requirement: Swin2D + Swin3D must stay frozen.
        # ============================================================
        vb = getattr(self.model, "video_backbone", None)
        ab = getattr(self.model, "audio_backbone", None)

        if vb is not None:
            vb.requires_grad_(False)
            vb.eval()

        if ab is not None:
            ab.requires_grad_(False)
            ab.eval()

        # ============================================================
        # [ADDED] Stage-2 head (Option B: attach head to model)
        # Reason:
        #   - Keeps optimizer wiring stable (head params are in self.model.parameters()).
        # ============================================================
        if not hasattr(self.model, "stage2_head"):
            k = None

            # Common inference paths to locate k
            if hasattr(self.model, "vacl") and hasattr(self.model.vacl, "k"):
                k = int(self.model.vacl.k)
            elif hasattr(self.model, "vacl_wrapper") and hasattr(self.model.vacl_wrapper, "vacl"):
                k = int(self.model.vacl_wrapper.vacl.k)
            elif hasattr(self.model, "vacl_wrapper") and hasattr(self.model.vacl_wrapper, "k"):
                k = int(self.model.vacl_wrapper.k)

            if k is None:
                raise RuntimeError("[AVPretrainSystem] Could not infer k for Stage2AVClassifierHead from model.")

            # ============================================================
            # ===================== [GRID SEARCH] Head Construction =====================
            # Description:
            #   Uses the head hyperparameters passed into AVPretrainSystem
            #   (grid-search controlled) while preserving defaults.
            # ==============================================================================
            mlp_hidden = None if int(self._stage2_mlp_hidden) < 0 else int(self._stage2_mlp_hidden)

            self.model.stage2_head = Stage2AVClassifierHead(
                k=k,
                cfg=Stage2HeadConfig(
                    num_classes=2,
                    pool=str(self._stage2_pool),
                    use_layernorm=bool(self._stage2_use_layernorm),
                    mlp_hidden=mlp_hidden,
                    dropout=float(self._stage2_dropout),
                ),
            )

        # ============================================================
        # [ADDED] Metrics (epoch aggregated, DDP-safe)
        # ============================================================
        self.train_acc = BinaryAccuracy()
        self.train_auc = BinaryAUROC(thresholds=self.val_auc_thresholds)
        self.train_f1 = BinaryF1Score()

        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC(thresholds=self.val_auc_thresholds)
        self.val_f1 = BinaryF1Score()

        # ============================================================
        # [ADDED] Inactive stubs (explicitly OFF)
        # - logits-native loss stub kept OFF
        # - BMN loss stub kept OFF
        # ============================================================
        self._USE_LOGITS_LOSS = False
        self._USE_BMN_LOSS = False

        # ============================================================
        # [KEPT] Checkpointing
        # ============================================================
        self.save_dir = Path("checkpoints_stage2")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._best_val = float("inf")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ============================================================
    # [ADDED] Ensure frozen Swin backbones stay in eval()
    # Reason: Lightning toggles train() globally; we want Swin dropout/stochdepth off.
    # ============================================================
    def on_train_epoch_start(self) -> None:
        vb = getattr(self.model, "video_backbone", None)
        ab = getattr(self.model, "audio_backbone", None)
        if vb is not None:
            vb.eval()
        if ab is not None:
            ab.eval()

    # ============================================================
    # [ADDED] Safe batch transfer (labels included)
    # ============================================================
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch["audio"] = batch["audio"].to(device, non_blocking=True)
        batch["video"] = batch["video_u8_cthw"].to(device, non_blocking=True).float().div_(255.0)

        # [ADDED] move labels if present
        for k in ("y", "label", "y_onehot", "label_onehot"):
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)

        return batch

    # ============================================================
    # [PATCHED] training_step (Stage-2)
    # ============================================================
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self.model(
            video_in=batch["video"],
            audio_in=batch["audio"].unsqueeze(1),
        )

        # ------------------------------------------------------------
        # [REQUIRED OUTPUTS]
        # ------------------------------------------------------------
        X_v_att = out["X_v_att"]
        X_a_att = out["X_a_att"]
        L_cor = out["L_cor"]
        l_infonce = out["l_infonce"]

        # ------------------------------------------------------------
        # [ADDED] Head forward
        # ------------------------------------------------------------
        head_out = self.model.stage2_head(X_v_att, X_a_att)
        P = head_out["P"]            # (B,2)
        logits = head_out["logits"]  # (B,2)

        # ------------------------------------------------------------
        # [ADDED] Resolve labels
        # Supports:
        #   y_onehot: (B,2)  OR  y: (B,)
        # ------------------------------------------------------------
        y_onehot = batch.get("y_onehot", batch.get("label_onehot"))
        y_class = batch.get("y", batch.get("label"))

        if y_onehot is None:
            if y_class is None:
                raise KeyError("[AVPretrainSystem] Missing labels: provide y/label or y_onehot/label_onehot.")
            y_class = y_class.long()
            y_onehot = torch.zeros((y_class.size(0), 2), device=y_class.device, dtype=P.dtype)
            y_onehot.scatter_(1, y_class[:, None], 1.0)
        else:
            y_onehot = y_onehot.to(device=P.device, dtype=P.dtype)
            y_class = y_onehot.argmax(dim=1).long()

        # ============================================================
        # [ACTIVE] Paper Eq.18 BCE on probabilities
        # ============================================================
        L_bce = bce_paper_eq18(P, y_onehot)
        L_cls = L_bce

        # ============================================================
        # [INACTIVE STUB] logits-native loss (OFF)
        # If enabled later, replace L_cls to avoid double counting.
        # ============================================================
        if self._USE_LOGITS_LOSS:
            L_cls = torch.nn.functional.cross_entropy(logits, y_class)

        # ============================================================
        # [INACTIVE STUB] L_bmn (OFF)
        # NOTE: only computed/logged if enabled.
        # ============================================================
        L_bmn = None
        if self._USE_BMN_LOSS:
            L_bmn = logits.new_zeros(())  # placeholder

        # ============================================================
        # [FINAL LOSS]
        # loss_total = omega*L_cor + lambda*l_infonce + beta*L_cls (+ alpha*L_bmn if enabled)
        # ============================================================
        loss_total = (self.omega * L_cor) + (self.lambda_ * l_infonce) + (self.beta * L_cls)
        if L_bmn is not None:
            loss_total = loss_total + (self.alpha * L_bmn)

        # ============================================================
        # [LOGS] (non-optionals as requested)
        # ============================================================
        self.log("train/loss_total", loss_total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/L_cor", L_cor, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/L_bce", L_bce, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/l_infonce", l_infonce, on_step=True, on_epoch=True, sync_dist=True)
        if L_bmn is not None:
            self.log("train/L_bmn", L_bmn, on_step=True, on_epoch=True, sync_dist=True)

        # ============================================================
        # [METRICS] ACC / AUC / F1 (epoch aggregated)
        # ============================================================
        score_pos = P[:, 1]
        self.train_acc(score_pos, y_class)
        self.train_auc(score_pos, y_class)
        self.train_f1(score_pos, y_class)

        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/auc", self.train_auc, on_epoch=True, sync_dist=True)
        self.log("train/f1", self.train_f1, on_epoch=True, sync_dist=True)

        return loss_total

    # ============================================================
    # [PATCHED] validation_step (Stage-2) + metric cap
    # ============================================================
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self.model(
            video_in=batch["video"],
            audio_in=batch["audio"].unsqueeze(1),
        )

        X_v_att = out["X_v_att"]
        X_a_att = out["X_a_att"]
        L_cor = out["L_cor"]
        l_infonce = out["l_infonce"]

        head_out = self.model.stage2_head(X_v_att, X_a_att)
        P = head_out["P"]
        logits = head_out["logits"]

        y_onehot = batch.get("y_onehot", batch.get("label_onehot"))
        y_class = batch.get("y", batch.get("label"))

        if y_onehot is None:
            if y_class is None:
                raise KeyError("[AVPretrainSystem] Missing labels: provide y/label or y_onehot/label_onehot.")
            y_class = y_class.long()
            y_onehot = torch.zeros((y_class.size(0), 2), device=y_class.device, dtype=P.dtype)
            y_onehot.scatter_(1, y_class[:, None], 1.0)
        else:
            y_onehot = y_onehot.to(device=P.device, dtype=P.dtype)
            y_class = y_onehot.argmax(dim=1).long()

        L_bce = bce_paper_eq18(P, y_onehot)
        L_cls = L_bce

        if self._USE_LOGITS_LOSS:
            L_cls = torch.nn.functional.cross_entropy(logits, y_class)

        L_bmn = None
        if self._USE_BMN_LOSS:
            L_bmn = logits.new_zeros(())

        loss_total = (self.omega * L_cor) + (self.lambda_ * l_infonce) + (self.beta * L_cls)
        if L_bmn is not None:
            loss_total = loss_total + (self.alpha * L_bmn)

        self.log("val/loss_total", loss_total, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/L_cor", L_cor, on_epoch=True, sync_dist=True)
        self.log("val/L_bce", L_bce, on_epoch=True, sync_dist=True)
        self.log("val/l_infonce", l_infonce, on_epoch=True, sync_dist=True)
        if L_bmn is not None:
            self.log("val/L_bmn", L_bmn, on_epoch=True, sync_dist=True)

        # ============================================================
        # [ADDED] Metric cap: update state only for first N val batches
        # ============================================================
        if batch_idx < self.val_metric_cap_batches:
            score_pos = P[:, 1]
            self.val_acc(score_pos, y_class)
            self.val_auc(score_pos, y_class)
            self.val_f1(score_pos, y_class)

        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/auc", self.val_auc, on_epoch=True, sync_dist=True)
        self.log("val/f1", self.val_f1, on_epoch=True, sync_dist=True)

        return loss_total

    # ============================================================
    # ===================== [GRID SEARCH] Optimizer Param Groups =====================
    # Description:
    #   Adds trailing param-groups:
    #     - Group 0: "other trainable params" (non-head)
    #     - Group 1: "stage2_head params"
    #   Defaults preserve your original single-group AdamW behavior.
    # ============================================================
    def configure_optimizers(self):
        head_params = []
        other_params = []

        head = getattr(self.model, "stage2_head", None)
        head_param_ids = set()

        if head is not None:
            for p in head.parameters():
                if p.requires_grad:
                    head_params.append(p)
                    head_param_ids.add(id(p))

        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in head_param_ids:
                continue
            other_params.append(p)

        param_groups = []

        if len(other_params) > 0:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": self.lr if self.lr_backbone is None else self.lr_backbone,
                    "weight_decay": self.weight_decay if self.weight_decay_backbone is None else self.weight_decay_backbone,
                }
            )

        if len(head_params) > 0:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": self.lr if self.lr_head is None else self.lr_head,
                    "weight_decay": self.weight_decay if self.weight_decay_head is None else self.weight_decay_head,
                }
            )

        if len(param_groups) == 0:
            raise RuntimeError("[AVPretrainSystem] No trainable parameters found for optimizer.")

        return torch.optim.AdamW(param_groups)



def _run_one(args: argparse.Namespace, exp_name_suffix: str = "") -> float:
    torch.set_float32_matmul_precision("high")

    offline_root = (REPO_ROOT / args.offline_root).resolve()
    tb_logdir = (REPO_ROOT / args.tb_logdir).resolve()
    tb_logdir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name if exp_name_suffix == "" else f"{args.run_name}_{exp_name_suffix}"
    logger = TensorBoardLogger(save_dir=str(tb_logdir), name=run_name)

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

        # ===================== [GRID SEARCH] Head hparams =====================
        stage2_pool=args.stage2_pool,
        stage2_use_layernorm=bool(args.stage2_use_layernorm),
        stage2_mlp_hidden=int(args.stage2_mlp_hidden),
        stage2_dropout=float(args.stage2_dropout),

        # ===================== [GRID SEARCH] Optimizer param-group knobs =====================
        lr_head=args.lr_head,
        weight_decay_head=args.weight_decay_head,
        lr_backbone=args.lr_backbone,
        weight_decay_backbone=args.weight_decay_backbone,
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

        # ===================== [GRID SEARCH] Speed controls =====================
        limit_train_batches=getattr(args, "limit_train_batches", 1.0),
        limit_val_batches=getattr(args, "limit_val_batches", 1.0),
    )

    trainer.fit(system, datamodule=dm)

    # Prefer val/loss_total if present
    metric = trainer.callback_metrics.get("val/loss_total")
    if metric is None:
        metric = trainer.callback_metrics.get("train/loss_total")

    if metric is None:
        raise RuntimeError("[GRID SEARCH] No suitable metric found in trainer.callback_metrics.")

    return float(metric.detach().cpu()) if torch.is_tensor(metric) else float(metric)


# ============================================================
# ===================== [GRID SEARCH] Single Trial Runner =====================
# Description:
#   Builds datamodule/model/system/trainer exactly like your original script.
#   Returns a scalar score (prefers val/loss_total) for selection.
# ============================================================
def main() -> None:
    args = parse_args()

    # ============================================================
    # ===================== [GRID SEARCH] Mode =====================
    # Description:
    #   Sequentially runs multiple trials (each trial is its own Lightning fit).
    #   Does NOT modify your underlying logic; only varies init-time hparams.
    # ============================================================
    if args.grid_search:
        import itertools
        import copy

        base = copy.deepcopy(args)

        # Stage-1 screening defaults (override only in grid mode)
        base.devices = int(args.grid_devices)
        base.max_epochs = int(args.grid_max_epochs)
        base.limit_train_batches = float(args.grid_limit_train_batches)
        base.limit_val_batches = float(args.grid_limit_val_batches)

        # Build grids (empty => singleton from base args)
        lr_grid = _parse_csv_list(args.grid_lr, float) if args.grid_lr else [float(base.lr)]
        wd_grid = _parse_csv_list(args.grid_weight_decay, float) if args.grid_weight_decay else [float(base.weight_decay)]
        omega_grid = _parse_csv_list(args.grid_omega, float) if args.grid_omega else [float(base.omega)]
        lambda_grid = _parse_csv_list(args.grid_lambda_, float) if args.grid_lambda_ else [float(base.lambda_)]
        beta_grid = _parse_csv_list(args.grid_beta, float) if args.grid_beta else [float(base.beta)]

        pool_grid = _parse_csv_list(args.grid_stage2_pool, str) if args.grid_stage2_pool else [str(base.stage2_pool)]
        ln_grid = _parse_csv_list(args.grid_stage2_use_layernorm, int) if args.grid_stage2_use_layernorm else [1 if base.stage2_use_layernorm else 0]
        mlp_grid = _parse_csv_list(args.grid_stage2_mlp_hidden, int) if args.grid_stage2_mlp_hidden else [int(base.stage2_mlp_hidden)]
        drop_grid = _parse_csv_list(args.grid_stage2_dropout, float) if args.grid_stage2_dropout else [float(base.stage2_dropout)]

        lrh_grid = _parse_csv_list(args.grid_lr_head, float) if args.grid_lr_head else [base.lr_head]
        wdh_grid = _parse_csv_list(args.grid_weight_decay_head, float) if args.grid_weight_decay_head else [base.weight_decay_head]
        lrb_grid = _parse_csv_list(args.grid_lr_backbone, float) if args.grid_lr_backbone else [base.lr_backbone]
        wdb_grid = _parse_csv_list(args.grid_weight_decay_backbone, float) if args.grid_weight_decay_backbone else [base.weight_decay_backbone]

        best = {"score": float("inf"), "desc": None}

        for (lr, wd, omega, lam, beta,
             pool, ln, mlp, drop,
             lrh, wdh, lrb, wdb) in itertools.product(
            lr_grid, wd_grid, omega_grid, lambda_grid, beta_grid,
            pool_grid, ln_grid, mlp_grid, drop_grid,
            lrh_grid, wdh_grid, lrb_grid, wdb_grid
        ):
            trial = copy.deepcopy(base)

            trial.lr = float(lr)
            trial.weight_decay = float(wd)
            trial.omega = float(omega)
            trial.lambda_ = float(lam)
            trial.beta = float(beta)

            trial.stage2_pool = str(pool)
            trial.stage2_use_layernorm = bool(int(ln))
            trial.stage2_mlp_hidden = int(mlp)
            trial.stage2_dropout = float(drop)

            trial.lr_head = lrh if lrh is None else float(lrh)
            trial.weight_decay_head = wdh if wdh is None else float(wdh)
            trial.lr_backbone = lrb if lrb is None else float(lrb)
            trial.weight_decay_backbone = wdb if wdb is None else float(wdb)

            suffix = (
                f"lr{trial.lr}_wd{trial.weight_decay}"
                f"_om{trial.omega}_lam{trial.lambda_}_b{trial.beta}"
                f"_pool{trial.stage2_pool}_ln{int(trial.stage2_use_layernorm)}"
                f"_mlp{trial.stage2_mlp_hidden}_dr{trial.stage2_dropout}"
                f"_lrh{trial.lr_head}_wdh{trial.weight_decay_head}"
                f"_lrb{trial.lr_backbone}_wdb{trial.weight_decay_backbone}"
            )

            score = _run_one(trial, exp_name_suffix=suffix)

            if score < best["score"]:
                best = {"score": score, "desc": suffix}

            torch.cuda.empty_cache()

        print(f"[GRID SEARCH DONE] best_score={best['score']:.6f} best_cfg={best['desc']}")
        return

    # ============================================================
    # [KEPT] Original single-run behavior
    # ============================================================
    torch.set_float32_matmul_precision("high")

    offline_root = (REPO_ROOT / args.offline_root).resolve()
    tb_logdir = (REPO_ROOT / args.tb_logdir).resolve()
    tb_logdir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=str(tb_logdir), name=args.run_name)

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

    audio_backbone = Swin2DAudioBackboneWrapper(Swin2DAudioWrapperConfig())
    audio_backbone.requires_grad_(False)
    audio_backbone.eval()

    swin3d_cfg = BuildSwin3DConfig(out="5d", use_checkpoint=True)
    video_backbone = VideoBackboneSwin3D(swin3d_cfg)
    video_backbone.requires_grad_(False)
    video_backbone.eval()

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

        # ===================== [GRID SEARCH] Head hparams (single-run too) =====================
        stage2_pool=args.stage2_pool,
        stage2_use_layernorm=bool(args.stage2_use_layernorm),
        stage2_mlp_hidden=int(args.stage2_mlp_hidden),
        stage2_dropout=float(args.stage2_dropout),

        # ===================== [GRID SEARCH] Optimizer param-group knobs =====================
        lr_head=args.lr_head,
        weight_decay_head=args.weight_decay_head,
        lr_backbone=args.lr_backbone,
        weight_decay_backbone=args.weight_decay_backbone,
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


# if __name__ == "__main__":
#     main()
#
#





