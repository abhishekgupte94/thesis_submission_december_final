# sanity_check_new_mvit_storehouse.py
import torch
from thesis_main_files.main_files.unused_old_work.feature_extraction import MViTv2FeatureExtractor

# pick device safely (works on CPU or CUDA boxes)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1) init (no logic changes)
ext = MViTv2FeatureExtractor(
    device=device,
    preserve_temporal=True,   # keep your temporal path
    temporal_pool=False,       # (T, D) after spatial pooling
    aggregate="mean",
    dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
    verbose=True,
    default_save_dir=None,
)

print("✅ extractor initialized on", device)



# test with a known video that has audio track
video_path = "/thesis_main_files/data/processed_files/lav_df/new_setup/evaluate_files/A_only_lt7p5/000035.mp4"


# 2) single file
res = ext.extract_one(video_path, save=False)

# 3) verify the classic dict layout (unchanged)
assert isinstance(res, dict), "Expected dict result"
assert {"path","features","shape"}.issubset(res.keys()), f"Missing keys: {res.keys()}"
print("path:", res["path"])
print("shape:", res["shape"])
print("tensor?", isinstance(res["features"], torch.Tensor), "dtype:", res["features"].dtype, "device:", res["features"].device)

# 4) OPTIONAL: saving smoke test
# ext.extract_one(video_path, save=True, save_dir="./mvit_out")  # writes <stem>.pt
