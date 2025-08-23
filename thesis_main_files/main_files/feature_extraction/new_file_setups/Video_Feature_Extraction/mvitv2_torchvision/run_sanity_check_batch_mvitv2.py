# run_sanity_check_video_batch_adapter.py
import torch
# from mvit_adapter import
from thesis_main_files.main_files.feature_extraction.new_file_setups.Video_Feature_Extraction.mvitv2_torchvision.mvit_adapter import MViTVideoFeatureExtractor

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
use_amp = torch.cuda.is_available()

# 1) init adapter (it builds the core extractor inside)
adapter = MViTVideoFeatureExtractor(
    device=device,
    amp=use_amp,
    strict_temporal=False,     # keep False to allow mixed T′ via time-mean
    save_video_feats=False,
    save_dir=None,
    preserve_temporal=True,
    temporal_pool=True,
    aggregate="none",
)
print(f"✅ adapter initialized on {device}")

# 2) small batch of paths
import os

video_dir = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/main_files/feature_extraction/new_file_setups/Audio_Feature_Extraction/ast_huggingface/video_dir_test"

valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
video_paths = [
    os.path.join(video_dir, f)
    for f in sorted(os.listdir(video_dir))
    if os.path.splitext(f)[1].lower() in valid_exts
]

print("First 8 video paths:")
print(video_paths[:8])

# 3) run
batch = adapter.execute(video_paths)  # -> (B, T′, D) if equal T′, else (B, D)
print("batch shape:", tuple(batch.shape), "| dtype:", batch.dtype, "| device:", batch.device)

# 4) quick checks
assert batch.shape[0] == len(video_paths), "B mismatch"
assert isinstance(batch, torch.Tensor)
print("✅ batch sanity OK")
