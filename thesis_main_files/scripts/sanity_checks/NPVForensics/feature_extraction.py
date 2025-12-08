import torch

# Import your wrapper
from scripts.feature_extraction.main.main_feature_extraction_wrapper import (
    _build_swin_model,
    _maybe_load_pretrained,
    JointAVModel
)

# ---------------------------------------------------------
# 1. Build the AUDIO Swin model
# ---------------------------------------------------------

audio_cfg = "configs/swin_audio.yaml"
audio_model, audio_config = _build_swin_model(audio_cfg)

# Optionally load pretrained ImageNet weights
audio_model = _maybe_load_pretrained(audio_model, audio_config)


# ---------------------------------------------------------
# 2. Build the VIDEO Swin model
# ---------------------------------------------------------

video_cfg = "configs/swin_video.yaml"
video_model, video_config = _build_swin_model(video_cfg)

video_model = _maybe_load_pretrained(video_model, video_config)


# ---------------------------------------------------------
# 3. Wrap them in your JointAVModel
# ---------------------------------------------------------

joint_model = JointAVModel(audio_model, video_model)
joint_model.eval()  # feature extraction mode


# ---------------------------------------------------------
# 4. Fake input tokens (for demonstration)
#     Shape: (B, N_tokens, D)
# ---------------------------------------------------------

B = 2

# Example: audio tokens from your audio tokenizer
audio_tokens = torch.randn(B, 50, 128)       # (batch=2, tokens=50, dim=128)

# Example: video tokens from your video tokenizer
video_tokens = torch.randn(B, 32, 384)       # (batch=2, tokens=32, dim=384)


# ---------------------------------------------------------
# 5. Run the forward pass
# ---------------------------------------------------------

with torch.no_grad():
    audio_encoded, video_encoded = joint_model(audio_tokens, video_tokens)

print("Audio Encoded:", audio_encoded.shape)
print("Video Encoded:", video_encoded.shape)
