# train.py — updated for TxD BatfdPlus and existing loss/utils stack

import argparse
import os

import toml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Dataset / DataModule
from lavdf import LavdfDataModule

# Models
# NOTE: Assume:
#   - model/batFDplus.py defines the *TxD* BatfdPlus (LightningModule)
#   - model/batfd.py      defines the baseline Batfd (LightningModule)
from batFDplus import BatfdPlus
# from model.batfd import Batfd

# Utilities (callbacks + metadata helper)
from utils import LrLogger, EarlyStoppingLR, generate_metadata_min


def parse_args():
    parser = argparse.ArgumentParser(description="BATFD / BATFD+ training (TxD-ready)")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to TOML config (e.g. configs/batfd_plus_lavdf.toml)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root path to LAV-DF data directory")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", default="32",
                        help="Precision: 32, 16, '16-mixed', etc.")
    parser.add_argument("--num_train", type=int, default=None,
                        help="Limit number of train samples (debug)")
    parser.add_argument("--num_val", type=int, default=1000,
                        help="Limit number of val samples")
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def main():
    args = parse_args()
    config = toml.load(args.config)

    # ------------------------------------------------------------------
    # 1) Ensure minimal metadata exists (used by LavdfDataModule)
    # ------------------------------------------------------------------
    metadata_min_path = os.path.join(args.data_root, "metadata.min.json")
    if not os.path.exists(metadata_min_path):
        generate_metadata_min(args.data_root)

    # ------------------------------------------------------------------
    # 2) Basic config unpacking
    # ------------------------------------------------------------------
    dataset_name = config.get("dataset", "lavdf")

    # Optimizer / LR scaling (linear scaling w.r.t batch size * gpus)
    base_lr = config["optimizer"]["learning_rate"]
    gpus = args.gpus
    effective_batch_size = args.batch_size * max(gpus, 1)
    learning_rate = base_lr * effective_batch_size / 4  # 4 = reference batch size

    num_frames = config["num_frames"]
    max_duration = config["max_duration"]

    # Model config
    model_cfg = config["model"]
    v_enc_cfg = model_cfg["video_encoder"]
    a_enc_cfg = model_cfg["audio_encoder"]
    cla_cfg = model_cfg["frame_classifier"]
    bm_cfg = model_cfg["boundary_module"]
    opt_cfg = config["optimizer"]

    model_type = config.get("model_type", "batfd_plus")  # "batfd_plus" or "batfd"

    # ------------------------------------------------------------------
    # 3) Build model (TxD BatfdPlus or baseline Batfd)
    # ------------------------------------------------------------------
    # if model_type == "batfd_plus":
        # TxD BatfdPlus — uses your new TxD blocks internally
    model = BatfdPlus(
        v_encoder=v_enc_cfg["type"],
        a_encoder=a_enc_cfg["type"],
        frame_classifier=cla_cfg["type"],
        ve_features=v_enc_cfg["hidden_dims"],
        ae_features=a_enc_cfg["hidden_dims"],
        v_cla_feature_in=v_enc_cfg["cla_feature_in"],
        a_cla_feature_in=a_enc_cfg["cla_feature_in"],
        boundary_features=bm_cfg["hidden_dims"],
        boundary_samples=bm_cfg["samples"],
        temporal_dim=num_frames,
        max_duration=max_duration,
        weight_frame_loss=opt_cfg["frame_loss_weight"],
        weight_modal_bm_loss=opt_cfg["modal_bm_loss_weight"],
        weight_contrastive_loss=opt_cfg["contrastive_loss_weight"],
        contrast_loss_margin=opt_cfg["contrastive_loss_margin"],
        cbg_feature_weight=opt_cfg["cbg_feature_weight"],
        prb_weight_forward=opt_cfg["prb_weight_forward"],
        weight_decay=opt_cfg["weight_decay"],
        learning_rate=learning_rate,
        distributed=args.gpus > 1,
    )
    require_match_scores = True   # BSN++ / BATFD+ needs match scores # Does not need scores if you are not using CBPG
    get_meta_attr = BatfdPlus.get_meta_attr

    # elif model_type == "batfd":
    #     # Baseline BATFD (no BSN++ extras)
    #     model = Batfd(
    #         v_encoder=v_enc_cfg["type"],
    #         a_encoder=a_enc_cfg["type"],
    #         frame_classifier=cla_cfg["type"],
    #         ve_features=v_enc_cfg["hidden_dims"],
    #         ae_features=a_enc_cfg["hidden_dims"],
    #         v_cla_feature_in=v_enc_cfg["cla_feature_in"],
    #         a_cla_feature_in=a_enc_cfg["cla_feature_in"],
    #         boundary_features=bm_cfg["hidden_dims"],
    #         boundary_samples=bm_cfg["samples"],
    #         temporal_dim=num_frames,
    #         max_duration=max_duration,
    #         weight_frame_loss=opt_cfg["frame_loss_weight"],
    #         weight_modal_bm_loss=opt_cfg["modal_bm_loss_weight"],
    #         weight_contrastive_loss=opt_cfg["contrastive_loss_weight"],
    #         contrast_loss_margin=opt_cfg["contrastive_loss_margin"],
    #         weight_decay=opt_cfg["weight_decay"],
    #         learning_rate=learning_rate,
    #         distributed=args.gpus > 1,
    #     )
    #     require_match_scores = False
    #     get_meta_attr = Batfd.get_meta_attr

    # else:
    #     raise ValueError(f"Invalid model_type in config: {model_type}")

    # ------------------------------------------------------------------
    # 4) DataModule for LAV-DF
    # ------------------------------------------------------------------
    if dataset_name.lower() != "lavdf":
        raise ValueError(f"Unsupported dataset: {dataset_name} (expected 'lavdf')")

    # NOTE: feature_types=(None, None) means "load raw audio/video and let model encode".
    # If you later precompute features, this is where you'd switch to ("video", "audio").
    v_feature_type = None
    a_feature_type = None

    dm = LavdfDataModule(
        root=args.data_root,
        frame_padding=num_frames,
        require_match_scores=require_match_scores,
        feature_types=(v_feature_type, a_feature_type),
        max_duration=max_duration,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        take_train=args.num_train,
        take_dev=args.num_val,
        get_meta_attr=get_meta_attr,
    )

    # ------------------------------------------------------------------
    # 5) Precision handling
    # ------------------------------------------------------------------
    precision = args.precision
    try:
        precision = int(precision)
    except ValueError:
        # keep as string for things like "16-mixed"
        pass

    # ------------------------------------------------------------------
    # 6) Trainer & callbacks
    # ------------------------------------------------------------------
    exp_name = config.get("name", "batfd_experiment")
    ckpt_dir = os.path.join("./ckpt", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    monitor_metric = "val_fusion_bm_loss" if model_type == "batfd_plus" else "val_bm_loss"

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        filename=exp_name + "-{epoch}-{" + monitor_metric + ":.3f}",
        monitor=monitor_metric,
        mode="min",
    )

    lr_logger_cb = LrLogger()
    early_stopping_lr_cb = EarlyStoppingLR(lr_threshold=1e-7)

    trainer = Trainer(
        log_every_n_steps=50,
        precision=precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb, lr_logger_cb, early_stopping_lr_cb],
        enable_checkpointing=True,
        benchmark=True,
        accelerator="auto",
        devices=args.gpus,
        strategy=None if args.gpus < 2 else "ddp",
        # For PL >=2, `ckpt_path` is used for resume; for PL<2 use resume_from_checkpoint
        default_root_dir=ckpt_dir,
    )

    # ------------------------------------------------------------------
    # 7) Fit
    # ------------------------------------------------------------------
    if args.resume:
        # Newer PL versions: trainer.fit(..., ckpt_path=...)
        trainer.fit(model, dm, ckpt_path=args.resume)
    else:
        trainer.fit(model, dm)


# if __name__ == "__main__":
#     main()
