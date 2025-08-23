# sanity_check_batch_ast.py (short)
import torch
# from extract_audio_features_from_AST import ASTAudioExtractor
from thesis_main_files.main_files.feature_extraction.new_file_setups.Audio_Feature_Extraction.ast_huggingface.extract_audio_features_from_AST import ASTAudioExtractor
import os
device = "cuda:0" if torch.cuda.is_available() else "cpu"
ext = ASTAudioExtractor(device=device, time_series=True)
video_dir = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/main_files/feature_extraction/new_file_setups/Audio_Feature_Extraction/ast_huggingface/video_dir_test"

valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
paths = [
    os.path.join(video_dir, f)
    for f in sorted(os.listdir(video_dir))
    if os.path.splitext(f)[1].lower() in valid_exts
]

print("First 8 video paths:")
print(paths[:8])

# paths = [
#     "/path/to/a.mp4", "/path/to/b.mp4",
#     "/path/to/c.mp4", "/path/to/d.mp4",
# ]
items = ext.extract_from_paths(paths, save=False)
feats = [r["features"] for r in items]
shapes = {tuple(f.shape) for f in feats}
if len(shapes) == 1:
    batch = torch.stack(feats, 0)
else:
    batch = torch.stack([f.mean(0) if f.dim()==2 else f for f in feats], 0)
print("audio batch:", batch.shape, batch.dtype, batch.device)
