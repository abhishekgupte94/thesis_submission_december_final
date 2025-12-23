import torch
from build_model  import BATFDInferenceWrapper

wrapper = BATFDInferenceWrapper(
    checkpoint_path="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/pretrained_weights/batfd_plus_default.ckpt",   # change this
    model_type="batfd_plus",
)

# IMPORTANT: dummy shapes must match what the model expects.
# If these shapes are wrong for your checkpoint, you'll get a shape error (good signal).
# Start small and adjust to match your actual batch tensors.
B = 1
T = 32  # placeholder
video = torch.randn(B, 3, T, 224, 224)   # common video layout (B,C,T,H,W)
audio = torch.randn(B, 1, 16000)         # placeholder waveform-ish

with torch.no_grad():
    out = wrapper(video, audio)

print(type(out))
print(len(out) if isinstance(out, (tuple, list)) else "not a tuple/list")
