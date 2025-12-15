# [NEW FILE] sanity_check_seq_token_encoders.py

from pathlib import Path

import torch

# [NEW BLOCK] Memory guard to protect your Mac
from utils.memory_guard.memory_guard import MemoryGuard

# [NEW BLOCK] Backbone + wrapper
from scripts.feature_extraction.SWIN.swin_feature_extraction_secondary_wrapper import (
    build_custom_swin_backbone,
    build_dummy_swin_config,
)
from scripts.feature_extraction.SWIN.main.main_feature_extraction_wrapper import AVSwinEncoder


# ----------------------------------------------------------------------
# Helper: safe .pt loader with shape checks
# ----------------------------------------------------------------------
def load_tokens_pt(path: str | Path) -> torch.Tensor:
    """
    Load a .pt file containing sequence tokens and normalise shape.

    Expected formats:
        - (B, S, D)
        - (S, D)   → auto-unsqueezed to (1, S, D)

    Raises if shape is not 2D or 3D.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[sanity] File not found: {path}")

    tensor = torch.load(path, map_location="cpu")

    if tensor.ndim == 2:
        # Assume (S, D) → treat as a single batch
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 3:
        raise ValueError(
            f"[sanity] Expected tensor with 2 or 3 dims, got shape {tuple(tensor.shape)}"
        )

    return tensor


# ----------------------------------------------------------------------
# Main sanity function
# ----------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Configure paths to your .pt files HERE.
    # These should be the outputs from your positional encoder (B, S, D).
    # ------------------------------------------------------------------
    AUDIO_PT_PATH = "PATH/TO/audio_tokens.pt"
    VIDEO_PT_PATH = "PATH/TO/video_tokens.pt"

    # ------------------------------------------------------------------
    # Memory guard setup
    # ------------------------------------------------------------------
    guard = MemoryGuard(
        max_process_gb=8.0,          # adjust if needed
        min_system_available_gb=2.0, # adjust if needed
        throws=True,
    )

    # Initial memory check before doing anything heavy
    guard.check()

    # ------------------------------------------------------------------
    # Load tokens from disk (CPU)
    # ------------------------------------------------------------------
    print("[sanity] Loading audio tokens from:", AUDIO_PT_PATH)
    audio_tokens = load_tokens_pt(AUDIO_PT_PATH)
    guard.check()

    print("[sanity] Loading video tokens from:", VIDEO_PT_PATH)
    video_tokens = load_tokens_pt(VIDEO_PT_PATH)
    guard.check()

    print(f"[sanity] Audio tokens shape: {tuple(audio_tokens.shape)}")
    print(f"[sanity] Video tokens shape: {tuple(video_tokens.shape)}")

    # ------------------------------------------------------------------
    # Device setup
    # ------------------------------------------------------------------
    # [UPDATED] safer device selection with CPU fallback
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[sanity] Using device: {device}")

    # ------------------------------------------------------------------
    # Build backbone + wrapper
    # For a *real* run, swap build_dummy_swin_config(...) with your
    # full Swin config object.
    # ------------------------------------------------------------------
    # [UPDATED] build_dummy_swin_config() now takes no args; num_classes goes into build_custom_swin_backbone
    num_classes = 10  # dummy for sanity; change for real experiment if needed
    config = build_dummy_swin_config()
    backbone = build_custom_swin_backbone(config=config, num_classes=num_classes)
    backbone.to(device)

    model = AVSwinEncoder(backbone=backbone)
    model.to(device)
    model.eval()

    # Move tokens to device
    audio_tokens = audio_tokens.to(device)
    video_tokens = video_tokens.to(device)

    # ------------------------------------------------------------------
    # Forward pass under no_grad + memory guard
    # ------------------------------------------------------------------
    # [NOTE] no_grad is used ONLY in this sanity script (outside the model) → no graph issues.
    with torch.no_grad():
        guard.check()
        # [UPDATED] explicitly request tokens + dict output to match new encoder API
        outputs = model(
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            return_tokens=True,
            return_dict=True,
        )

    # ------------------------------------------------------------------
    # Print resulting shapes
    # ------------------------------------------------------------------
    vid = outputs["video"]
    aud = outputs["audio"]

    print("\n[sanity] === VIDEO ENCODER OUTPUTS ===")
    print("  logits   :", None if vid["logits"] is None else tuple(vid["logits"].shape))
    print("  features :", None if vid["features"] is None else tuple(vid["features"].shape))
    print("  tokens   :", None if vid.get("tokens") is None else tuple(vid["tokens"].shape))

    print("\n[sanity] === AUDIO ENCODER OUTPUTS ===")
    print("  logits   :", None if aud["logits"] is None else tuple(aud["logits"].shape))
    print("  features :", None if aud["features"] is None else tuple(aud["features"].shape))
    print("  tokens   :", None if aud.get("tokens") is None else tuple(aud["tokens"].shape))

    print("\n[sanity] All good — sequence token encoders ran successfully.")


if __name__ == "__main__":
    try:
        main()
    except MemoryError as e:
        # If MemoryGuard trips, we land here instead of a hard crash.
        print(str(e))
        print("[sanity_check_seq_token_encoders] Exiting due to memory guard.")
