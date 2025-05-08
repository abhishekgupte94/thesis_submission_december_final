import torch
from torch.utils.data import Dataset

from pathlib import Path
import pandas as pd
from thesis_main_files.models.data_loaders.data_loader_ART import get_project_root
def create_file_paths_for_svm(project_dir_curr, csv_name="training_data_svm_final.csv"):
    """
    Generates full paths for lips-only and original videos for SVM training data,
    based on filenames from a CSV file.

    Args:
        project_dir_curr (Path or str): Base project directory.
        csv_name (str): Name of CSV file containing file listings and labels.

    Returns:
        tuple: lips_only_paths, original_paths, labels
    """
    project_dir_curr = Path(project_dir_curr)

    # Define paths for SVM training data
    csv_path = project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "final_training_data_svm" / csv_name
    csv_dir = project_dir_curr / "datasets" / "processed" / "lav_df" / "checks" / "data_to_preprocess_for_svm"

    # Swin Transformer project-specific paths
    project_dir_video_swin = Path(get_project_root("Video-Swin-Transformer"))
    video_dir = project_dir_video_swin / "data" / "train" / "real"

    # Read CSV and extract file paths
    df = pd.read_csv(csv_path)

    lips_only_paths = []
    original_paths = []
    for filename in df['video_file']:
        original_path = Path(filename)
        new_filename = original_path.stem + "_lips_only" + original_path.suffix
        full_lips_only_path = video_dir / new_filename
        full_original_path = csv_dir / original_path

        lips_only_paths.append(full_lips_only_path)
        original_paths.append(full_original_path)

    labels = df['label'].tolist()

    return lips_only_paths, original_paths, labels


class AudioVideoPathDataset(Dataset):
    """
    Custom PyTorch Dataset for loading lips-only video paths, original video paths, and labels
    specifically for SVM training.
    """
    def __init__(self, project_dir_curr, csv_name="training_data_svm_final.csv", augmentations=None):
        self.project_dir_curr = project_dir_curr
        self.csv_name = csv_name
        self.augmentations = augmentations

        # Load paths and labels for SVM training
        self.video_paths, self.audio_paths, self.labels = create_file_paths_for_svm(project_dir_curr, csv_name)

        # Store data as list of tuples
        self.data = list(zip(self.video_paths, self.audio_paths, self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, audio_path, label = self.data[idx]

        # Apply optional augmentations to video path
        if self.augmentations:
            video_path = self.augmentations(video_path)

        return str(audio_path), str(video_path), label
