from torch.utils.data import Dataset
# from torch.utils.tensorboard import SummaryWriter

# Importing required modules for video/audio preprocessing and feature extraction

import pandas as pd

def preprocess_videos_before_training(csv_path, csv_column, output_dir, batch_size=128):
    """
    Reads video paths from a CSV file and preprocesses them into a common output directory.

    Args:
        csv_path (str): Path to the CSV file.
        csv_column (str): Column in CSV containing video paths.
        output_dir (str): Directory where lip-only video will be saved.
        batch_size (int): Number of frames per batch for lip extraction.
    """
    project_dir_curr = get_project_root()
    csv_name = Path(csv_path).name
    _,video_paths,_ = create_file_paths(project_dir_curr,csv_name = csv_name)
    # Step 1: Read CSV
    # df = pd.read_csv(csv_path)
    # if csv_column not in df.columns:
    #     raise ValueError(f"Column '{csv_column}' not found in {csv_path}")
    #
    # video_paths = df[csv_column].tolist()

    # # Step 2: Initialize Preprocessor
    # preproc = VideoPreprocessor_FANET(
    #     batch_size=batch_size,
    #     output_base_dir=output_dir,
    #     device="cuda" # auto-handled per rank
    #     # use_fp16=True
    #
    # )
    parallel_main(video_paths,batch_size,output_dir)
    # Step 3: Preprocess all video
    # preproc.parallel_main(video_paths)

    print(f"âœ… All video preprocessed and saved to: {output_dir}")




def create_file_paths(project_dir_curr, csv_name="training_data_two.csv"):
    """
    Generates full paths for video files based on filenames from a CSV file,
    appending '_lips_only' to each filename before the extension.

    Args:
        project_dir_curr (Path or str): Base project directory.
        csv_name (str): Name of CSV file with file listings and labels.

    Returns:
        tuple: lips_only_paths, original_paths, labels
    """
    project_dir_curr = Path(project_dir_curr)

    # CSV and video directory paths
    csv_path = project_dir_curr / "data" / "processed_files" / "csv" / "lav_df" / "training_data" / csv_name
    csv_dir = project_dir_curr / "data" / "processed_files" /  "lav_df" / "train"
    df = pd.read_csv(csv_path)
    project_dir_curr = Path("Video-Swin-Transformer")
    video_dir = project_dir_curr / "data" / "train" / "real"

    # Read CSV and extract file paths


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

def create_file_paths_for_train(csv_path, video_dir):
    """
    Generates full paths for video files based on filenames from a CSV file.

    Args:
        csv_path (str or Path): Path to the CSV file containing filenames and labels.
        video_dir (str or Path): Directory where the corresponding video files are stored
                                 (from convert_paths_for_training).
        limit (int, optional): Maximum number of samples to return. Defaults to 5000.

    Returns:
        tuple: (original_paths, labels) limited to 'limit' entries.
    """
    csv_path = Path(csv_path)
    video_dir = Path(video_dir)

    df = pd.read_csv(csv_path)

    original_paths = []
    for filename in df['file']:
        full_original_path = video_dir / filename
        original_paths.append(str(full_original_path))

    labels = df['label'].tolist()

    # Restrict to the first 'limit' entries
    return original_paths[:3000], labels[:3000]






def get_project_root(project_name=None):
    """
    Locate the root directory of the project based on script path.

    Args:
        project_name (str, optional): Specific project name to locate.

    Returns:
        Path or None: Root directory if found, else None.
    """
    import os
    current = Path(os.getcwd()).resolve()

    # Look for the known parent folder
    for parent in current.parents:
        if parent.name == "thesis_main_files":
            base_dir = parent.parent
            break
    else:
        return None

    if project_name:
        # Return the matching subdirectory if it exists
        target_path = base_dir / project_name
        if target_path.exists() and target_path.is_dir():
            return target_path
        else:
            return None
    else:
        # Fallback search for common project directories
        project_names = {"thesis_main_files", "Video-Swin-Transformer", "melodyExtraction_JDC"}
        for parent in current.parents:
            if parent.name in project_names:
                return parent
    return None




def convert_paths():
    """
    Prepare all necessary paths for processing and feature extraction.

    Returns:
        Tuple containing all path strings used for video preprocessing and feature extraction.
    """
    # project_dir_curr = Path("/content/project_combined_repo_clean/thesis_main_files")
    project_dir_curr = get_project_root()

    # Construct paths used in processing
    csv_path = str(project_dir_curr / "data" / "processed_files" / "csv" / "lav_df" / "training_data" / "training_data_two.csv")
    video_dir = str(project_dir_curr / "data" / "processed_files" / "lav_df" / "train")

    # Swin Transformer project-specific paths
    project_dir_video_swin = get_project_root("Video-Swin-Transformer")
    video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
    real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
    feature_dir_vid = str(project_dir_video_swin)

    return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path
from pathlib import Path

def convert_paths_for_training(csv_name: str = "sample_real_70_percent_half1.csv"):
    """
    Prepare all necessary paths for processing and feature extraction (training).
    CSV file stays with .csv extension, but the video directory uses the filename without extension.
    """
    project_dir_curr = get_project_root()

    # Build paths
    csv_path = str(
        project_dir_curr / "data" / "processed_files" / "csv"
        / "lav_df" / "new_setup" / "train_files" / csv_name
    )

    video_dir_name = Path(csv_name).stem  # remove .csv extension
    video_dir = str(
        project_dir_curr / "data" / "processed_files" / "lav_df"
        / "new_setup" / "train_files" / video_dir_name
    )

    return csv_path, video_dir




###############################################################################
# COMPONENT + FEATURE EXTRACTION CLASSES (Audio restored via video paths)
###############################################################################

# class VideoAudioFeatureExtractor:
#     """
#     Responsible for feature extraction from preprocessed video components and audio waveforms.
#     """
#     def __init__(self, device=None, amp=True, save_audio_feats=False, audio_save_dir=None):
#         self.device = device
#         self.mvit_adapter = MViTVideoFeatureExtractor(
#             device=self.device,  # torch.device(f"cuda:{local_rank}")
#             amp=True,  # uses fp16 autocast in your _forward_model
#             strict_temporal=False,  # set True to enforce equal T' within a batch
#             save_video_feats=False,  # set True if you want .pt saved per sample
#             save_dir=None,  # or a path to store .pt files
#             preserve_temporal=True,
#             temporal_pool=True,
#             aggregate="mean"
#         )
#         self.audio_extractor = ASTAudioExtractor(
#             device=self.device,
#             amp=amp,
#             time_series=True,     # 'yes' by default
#             token_pool="none",    # keep time series, no pooling
#             verbose=False,
#             default_save_dir=(audio_save_dir if save_audio_feats else None),
#         )
#     def extract_video_features(self, video_paths):
#         try:
#             # our MViT adapter exposes .execute(video_paths)
#             features = self.mvit_adapter.execute(video_paths)
#             return features
#         except Exception as e:
#             print(f"[VideoFeat] Error on {len(video_paths)} paths, e.g. {video_paths[:3]}... | {type(e).__name__}: {e}")
#             return None
#
#     def extract_audio_features(self, video_paths, batch_size,save_path = None):
#         try:
#             items = self.audio_extractor.extract_from_paths(
#                 video_paths,
#                 save=(self.audio_extractor.default_save_dir is not None),
#                 save_dir=self.audio_extractor.default_save_dir,
#                 overwrite=False
#             )
#             feats = [it["features"] for it in items]  # CPU tensors
#             shapes = {tuple(f.shape) for f in feats}
#             if len(shapes) == 1:
#                 batch = torch.stack(feats, dim=0)
#             else:
#                 batch = torch.stack([f.mean(0) if f.dim() == 2 else f for f in feats], dim=0)
#             # put on the same device as the extractor (trainerâ€™s rank)
#             return batch.to(self.audio_extractor.device, non_blocking=True)
#         except Exception as e:
#             print(f"[AudioFeat] Error on {len(video_paths)} paths, e.g. {video_paths[:3]}... | {type(e).__name__}: {e}")
#             return None

# # class VideoAudioFeatureProcessor:
#     """
#     Combines component and feature extractors to produce a usable dataset.
#     """
#     def __init__(self,batch_size,local_rank):
#         # self.video_preprocess_dir = video_preprocess_dir
#         # self.feature_dir_vid = feature_dir_vid
#
#         # # Initialize the video preprocessor (FANET)
#         # self.video_preprocessor = VideoPreprocessor_PIPNet(
#         #     # batch_size=batch_size,
#         #     output_base_dir_real=video_save_dir,
#         #     real_output_txt_path=output_txt_file
#         # )
#
#         # Initialize audio preprocessor
#         # self.audio_preprocessor = AudioPreprocessor()
#
#         # Initialize feature extractor (Swin Transformer)
#         # self.video_feature_ext = VideoAudioFeatureExtractor()
#         # self.video_feature_ext = mvit_extractor
#         # self.component_extractor = VideoComponentExtractor()
#         # torch.cuda.set_device(local_rank)
#         self.device = torch.device(f"cuda:{local_rank}")
#         self.feature_extractor = VideoAudioFeatureExtractor(device = self.device)  # pass rank device
#         self.batch_size = batch_size
#
#     def create_datasubset(self, csv_path, use_preprocessed=True, video_paths=None, audio_paths = None, video_save_dir=None, output_txt_file=None):
#         processed_video_features = None
#         processed_audio_features = None
#         video_error = False
#         audio_error = False
#
#         # try:
#         #     # Extract components from raw video
#         #     preprocessed_video_paths = self.component_extractor.extract_video_components(
#         #         video_paths, video_save_dir, output_txt_file, self.batch_size, self.video_preprocessor)
#         # except Exception as e:
#         #     print(f"Video Component Extraction Error: {e}")
#         #     video_error = True
#
#         try:
#             # Extract features from video if component extraction succeeded
#             # if not video_error:
#             processed_video_features = self.feature_extractor.extract_video_features(video_paths)
#         except Exception as e:
#             print(f"Video Feature Extraction Error: {e}")
#             video_error = True
#
#         try:
#             # Extract audio features using video file paths
#             processed_audio_features = self.feature_extractor.extract_audio_features(video_paths, self.batch_size)
#         except Exception as e:
#             print(f"Audio Feature Extraction Error: {e}")
#             audio_error = True
#
#         # Return features only if both succeeded
#         if not audio_error and not video_error:
#             return processed_audio_features, processed_video_features
#         else:
#             print("Errors encountered. No features returned.")
#             return None, None
#
###############################################################################
# DATASET CLASS
###############################################################################



class VideoAudioDataset(Dataset):
    """
    Dataset for loading video/audio samples for training.
    Expects a CSV and a video directory, then builds full file paths.
    """

    def __init__(self, csv_path: str, video_dir: str, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file with 'file' and 'label' columns.
            video_dir (str): Directory containing the corresponding video files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv_path = Path(csv_path)
        self.video_dir = Path(video_dir)
        self.transform = transform

        # Use helper to build paths + labels
        self.video_paths, self.labels = create_file_paths_for_train(self.csv_path, self.video_dir)

        assert len(self.video_paths) == len(self.labels), (
            f"Mismatch between number of files ({len(self.video_paths)}) "
            f"and labels ({len(self.labels)}). Check {self.csv_path}"
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        sample = {
            "video_path": str(video_path),
            "label": label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


