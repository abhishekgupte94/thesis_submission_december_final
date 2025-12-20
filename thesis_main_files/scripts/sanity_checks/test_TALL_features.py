"""
test_features.py

Simplified, hard-coded sanity test for TALL-Swin feature extraction.

Goal:
    - Run a quick sanity test on *up to 5 video* using your existing
      data pipeline and customised tall_swin model.
    - Verify that:
        * the dataloader works
        * the model forward pass works
        * logits and token features can be extracted and averaged
          to video level
    - Provide placeholders for evaluation metrics to be filled in later
      by a dedicated evaluator script.

This script *does not* use argparse; instead, you set constants in the
"USER CONFIGURATION" section below.
"""

import os
from typing import Dict, Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# --- Import your project modules (same style as main.py / engine.py) ---
from timm.models import create_model

import utils              # for (optionally) init_distributed_mode, etc.

from core.NPVForensics.custom_backbones.video_dataset import VideoDataSet
from core.NPVForensics.custom_backbones.video_dataset_aug import get_augmentor, build_dataflow
from core.NPVForensics.custom_backbones.video_dataset_config import get_dataset_config

# ============================================================
# USER CONFIGURATION (EDIT THESE VALUES)
# ============================================================

# Root directory where your video frames (or images) are stored.
DATA_DIR = "/path/to/frames_root"  # TODO: set this

# Directory containing the dataset list files (train/val/test txt or lmdb lists).
DATA_TXT_DIR = "/path/to/data_txt_dir"  # TODO: set this

# Which dataset config to use (must exist in DATASET_CONFIG).
DATASET_KEY = "ffpp"  # e.g., "ffpp"

# If you want to override which split/list file is used, set this.
# Otherwise, we will use the "test_list_name" from get_dataset_config.
VIDEO_LIST_PATH_OVERRIDE = ""  # e.g., "/path/to/custom_list.txt"

# Optional: path to a trained checkpoint (.pth) to load.
CHECKPOINT_PATH = ""  # leave empty to test with random weights

# Number of video to process for this sanity test.
MAX_VIDEOS = 5

# Basic model / input configuration (aligned with main.py defaults)
MODEL_NAME = "TALL_SWIN"
PRETRAINED = False       # Whether to use timm's pretrained weights (if any)
DURATION = 8             # num_groups in VideoDataSet
FRAMES_PER_GROUP = 1     # as in VideoDataSet; 1 means each "group" is 1 frame
NUM_CLIPS = 1            # number of clips per video
NUM_CROPS = 1            # number of crops (e.g., 1 for center crop)
DENSE_SAMPLING = True    # whether to use dense sampling in VideoDataSet
MODALITY = "rgb"         # VideoDataSet modality

INPUT_SIZE = 224         # input spatial size to the model
USE_LMDB = False         # whether the list files refer to LMDB
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TALL-Swin specific flags (these should mirror your main.py training setup)
HPE_TO_TOKEN = True      # hub position embedding to tokens
REL_POS = True           # relative positional encoding
WINDOW_SIZE = 14         # TALL-Swin window size
THUMBNAIL_ROWS = 4       # rows in TALL montage
TOKEN_MASK = True        # whether token masking inside model is enabled
DROP_RATE = 0.0
DROP_PATH_RATE = 0.1
DROP_BLOCK_RATE = 0.0
USE_CHECKPOINT = False   # gradient checkpointing in the model

# Autocast (mixed precision) for faster evaluation_for_detection_model on GPU
USE_AMP = True


# ============================================================
# PLACEHOLDER METRIC FUNCTIONS
# ============================================================

def compute_metrics_placeholder(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Placeholder function for evaluation metrics.

    - This is intentionally *simple* right now.
    - Later, you will replace this with calls into your dedicated
      evaluator script (e.g., metrics_evaluators.py).

    For now, we only print a few basic details and return an empty dict.
    """
    print("\n[METRICS PLACEHOLDER]")
    print(f"  logits shape : {logits.shape}  (N, num_classes)")
    print(f"  labels shape : {labels.shape}  (N,)")
    print("  TODO: plug in real metric computation here "
          "(accuracy, ROC-AUC, PR-AUC, etc.)")

    # You can add a trivial metric just to see something:
    # For example, how many times argmax(logits) == labels?
    preds = logits.argmax(axis=1)
    num_correct = (preds == labels).sum()
    acc = num_correct / float(labels.shape[0])
    print(f"  (Sanity) Accuracy from argmax logits: {acc:.4f}")

    return {
        "sanity_accuracy": float(acc),
        # Add real metrics here later.
    }


def print_metrics_placeholder(metrics: Dict[str, Any]) -> None:
    """
    Placeholder pretty-printer for metrics.

    This will eventually be replaced by a nicer printer from the
    evaluator script. For now, it just prints the dictionary.
    """
    print("\n[METRICS SUMMARY (PLACEHOLDER)]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


# ============================================================
# MODEL & DATALOADER BUILDING
# ============================================================

def build_model(num_classes: int) -> torch.nn.Module:
    """
    Construct the TALL-Swin model for evaluation_for_detection_model and move it to DEVICE.

    This mirrors the creation logic from main.py, but in a simplified,
    hard-coded form using the constants defined above. It also optionally
    loads a checkpoint from CHECKPOINT_PATH if provided.
    """
    # Decide number of input channels based on modality
    if MODALITY == "rgb":
        input_channels = 3
    elif MODALITY == "flow":
        # Example: 2 channels × 5 consecutive frames
        input_channels = 2 * 5
    else:
        # For sound / other cases, your project might handle channels
        # differently; here we keep 3 as a generic default.
        input_channels = 3

    print(f"Creating model: {MODEL_NAME}")
    print(f"  num_classes   : {num_classes}")
    print(f"  duration      : {DURATION}")
    print(f"  input_channels: {input_channels}")

    model = create_model(
        MODEL_NAME,
        pretrained=PRETRAINED,
        duration=DURATION,
        hpe_to_token=HPE_TO_TOKEN,
        rel_pos=REL_POS,
        window_size=WINDOW_SIZE,
        thumbnail_rows=THUMBNAIL_ROWS,
        token_mask=TOKEN_MASK,
        online_learning=False,
        num_classes=num_classes,
        drop_rate=DROP_RATE,
        drop_path_rate=DROP_PATH_RATE,
        drop_block_rate=DROP_BLOCK_RATE,
        use_checkpoint=USE_CHECKPOINT,
    )

    device = torch.device(DEVICE)
    model.to(device)

    # Load checkpoint if provided
    if CHECKPOINT_PATH:
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        if CHECKPOINT_PATH.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                CHECKPOINT_PATH, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

        # If checkpoint has a "model" key (common in your training code),
        # use utils.load_checkpoint; otherwise load directly.
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            print("  Found 'model' key in checkpoint; using utils.load_checkpoint")
            utils.load_checkpoint(model, checkpoint["model"])
        else:
            print("  Loading checkpoint state_dict directly into model")
            model.load_state_dict(checkpoint)

    model.eval()
    return model


def build_dataloader(
    num_classes: int,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader for evaluation, reusing:

        - VideoDataSet from video_dataset.py
        - get_augmentor / build_dataflow from video_dataset_aug.py
        - get_dataset_config from video_dataset_config.py

    We keep this aligned with your training/validation pipeline to ensure
    the preprocessing + sampling logic is identical for this sanity test.
    """
    # Obtain dataset-specific config (same as main.py)
    (
        cfg_num_classes,
        train_list_name,
        val_list_name,
        test_list_name,
        filename_seperator,
        image_tmpl,
        filter_video,
        label_file,
    ) = get_dataset_config(DATASET_KEY, USE_LMDB)

    assert cfg_num_classes == num_classes, (
        f"num_classes mismatch: {cfg_num_classes} (config) vs {num_classes} (arg)"
    )

    # Decide which list file to use for this sanity test.
    if VIDEO_LIST_PATH_OVERRIDE:
        list_path = VIDEO_LIST_PATH_OVERRIDE
        print(f"Using VIDEO_LIST_PATH_OVERRIDE: {list_path}")
    else:
        # Use the test list from the dataset config
        list_path = os.path.join(DATA_TXT_DIR, test_list_name)
        print(f"Using test list from dataset config: {list_path}")

    # Standard mean/std as in your augmentor (can tweak if needed)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # Evaluation-style augmentor:
    #   - is_train=False
    #   - cut_out=False (we don't want training augmentations in a test script)
    val_augmentor = get_augmentor(
        is_train=False,
        image_size=INPUT_SIZE,
        mean=list(mean),
        std=list(std),
        disable_scaleup=DENSE_SAMPLING,  # heuristic; mirrors your setup
        threed_data=False,
        version="v1",
        scale_range=None,
        modality=MODALITY,
        num_clips=NUM_CLIPS,
        num_crops=NUM_CROPS,
        cut_out=False,
        dataset=DATASET_KEY,
    )

    dataset = VideoDataSet(
        root_path=DATA_DIR,
        list_file=list_path,
        num_groups=DURATION,
        frames_per_group=FRAMES_PER_GROUP,
        num_clips=NUM_CLIPS,
        modality=MODALITY,
        dense_sampling=DENSE_SAMPLING,
        fixed_offset=True,
        image_tmpl=image_tmpl,
        transform=val_augmentor,
        is_train=False,
        test_mode=False,
        seperator=filename_seperator,
        filter_video=filter_video,
        num_classes=num_classes,
    )

    # Use build_dataflow to keep DataLoader settings consistent
    data_loader = build_dataflow(
        dataset=dataset,
        is_train=False,
        batch_size=2,   # small batch is fine for sanity
        workers=4,
        is_distributed=False,
    )

    return data_loader


# ============================================================
# MAIN SANITY-TEST LOGIC
# ============================================================

def run_sanity_test() -> None:
    """
    Run a simple sanity test:

        - Build model + dataloader
        - Process up to MAX_VIDEOS video
        - Extract:
            * video-level logits
            * video-level token features (mean over clips/crops)
        - Print shapes and basic info
        - Call placeholder metric functions
    """
    # (Optional) distributed init; safe even if run single-process
    utils.init_distributed_mode(None)

    print("=== TALL-Swin Sanity Test (Feature Extraction) ===")

    # Fix seeds and cuDNN settings for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Get dataset config to know num_classes
    num_classes, *_ = get_dataset_config(DATASET_KEY, USE_LMDB)

    # Build model and dataloader
    model = build_model(num_classes=num_classes)
    data_loader = build_dataloader(num_classes=num_classes)

    device = torch.device(DEVICE)

    all_video_logits = []      # list of [B, num_classes]
    all_video_labels = []      # list of [B]
    all_video_tokens = []      # list of [B, L, C]

    processed_videos = 0
    num_clips_times_crops = NUM_CLIPS * NUM_CROPS

    print("Beginning feature extraction loop...")

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_size = images.shape[0]

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            # Your customised tall_swin.py returns (logits, token_sequence).
            # The original version would just return logits. We handle both.
            model_output = model(images)

        if isinstance(model_output, (tuple, list)):
            logits, token_sequence = model_output
        else:
            logits = model_output
            token_sequence = None

        # Reshape logits: [B * (clips*crops), C] -> [B, clips*crops, C]
        logits = logits.reshape(batch_size, num_clips_times_crops, -1)
        # Average over clips/crops -> [B, num_classes]
        logits_video = logits.mean(dim=1)

        # Process token features if available
        if token_sequence is not None:
            # token_sequence: [B * (clips*crops), L, C]
            B_times_clips, L, C = token_sequence.shape
            assert B_times_clips == batch_size * num_clips_times_crops, (
                f"Unexpected token_sequence shape: {token_sequence.shape}, "
                f"expected first dim = batch_size * num_clips_times_crops "
                f"= {batch_size} * {num_clips_times_crops}"
            )
            tokens = token_sequence.reshape(batch_size, num_clips_times_crops, L, C)
            tokens_video = tokens.mean(dim=1)  # [B, L, C]
        else:
            tokens_video = None

        # Move to CPU and store
        all_video_logits.append(logits_video.detach().cpu())
        all_video_labels.append(labels.detach().cpu())
        if tokens_video is not None:
            all_video_tokens.append(tokens_video.detach().cpu())

        processed_videos += batch_size
        print(f"Processed video so far: {processed_videos}")

        if processed_videos >= MAX_VIDEOS:
            print(f"Reached MAX_VIDEOS = {MAX_VIDEOS}, stopping loop.")
            break

    # Stack results
    logits_arr = torch.cat(all_video_logits, dim=0).numpy()
    labels_arr = torch.cat(all_video_labels, dim=0).numpy()

    if all_video_tokens:
        tokens_arr = torch.cat(all_video_tokens, dim=0).numpy()
    else:
        tokens_arr = None

    print("\n=== SANITY TEST OUTPUT SHAPES ===")
    print(f"logits_arr shape : {logits_arr.shape}  (N, num_classes)")
    print(f"labels_arr shape : {labels_arr.shape}  (N,)")

    if tokens_arr is not None:
        print(f"tokens_arr shape : {tokens_arr.shape}  (N, L, C)")
    else:
        print("tokens_arr       : None (model did not return token features)")

    # Call placeholder metric computation
    metrics = compute_metrics_placeholder(logits_arr, labels_arr)
    print_metrics_placeholder(metrics)

    # (Optional) you can also save a tiny .npz here if you want
    # to inspect in a notebook. Comment out if not needed.
    out_path = "sanity_features.npz"
    save_dict: Dict[str, Any] = {
        "logits": logits_arr,
        "labels": labels_arr,
        "metrics": metrics,
    }
    if tokens_arr is not None:
        save_dict["token_features_mean"] = tokens_arr
    np.savez(out_path, **save_dict)
    print(f"\nSaved sanity-test features to {out_path}")


if __name__ == "__main__":
    run_sanity_test()
