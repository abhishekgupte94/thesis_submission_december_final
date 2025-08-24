# #
# from torch.utils.data import Dataset
# import torch
# from torch.utils.tensorboard.summary import video
# # from torch.utils.tensorboard import SummaryWriter
#
# # Importing required modules for video/audio preprocessing and feature extraction
# # from thesis_main_files.main_files.preprocessing.art_avdf.art.video_preprocessor_SAVE_FILES_MULTI_GPU import  parallel_main
# from thesis_main_files.main_files.feature_extraction.new_file_setups.Video_Feature_Extraction.mvitv2_torchvision.mvit_adapter import MViTVideoFeatureExtractor
# # from thesis_main_files.main_files.feature_extraction.art_avdf.art.feature_extractor_ART_Video import SWIN_EXECUTOR as VideoFeatureExtractor
# # from thesis_main_files.main_files.preprocessing.art_avdf.art.audio_preprocessorart import AudioPreprocessor
# # from thesis_main_files.main_files.preprocessing.art_avdf.art.video_preprocessorart_Fanet_gpu import VideoPreprocessor_FANET
# # from thesis_main_files.main_files.preprocessing.art_avdf.art.video_preprocessor_SAVE_FILES_MULTI_GPU import parallel_main
# from thesis_main_files.main_files.feature_extraction.new_file_setups.Audio_Feature_Extraction.ast_huggingface.extract_audio_features_from_AST import ASTAudioExtractor
#
# from pathlib import Path
# import pandas as pd
#
# # from video_preprocessor_fanet_multi_gpu import VideoPreprocessor_FANET
# def preprocess_videos_before_evaluation(csv_path, csv_column, output_dir, batch_size=128):
#     """
#     Reads video paths from a CSV file and preprocesses them into a common output directory.
#
#     Args:
#         csv_path (str): Path to the CSV file.
#         csv_column (str): Column CSV containing video paths.
#         output_dir (str): Directory where lip-only videos will be saved.
#         batch_size (int): Number of frames per batch for lip extraction.
#     """
#     import pandas as pd
#     project_dir_curr = get_project_root()
#     csv_name = Path(csv_path).name
#     _,video_paths,_ = create_file_paths(project_dir_curr,csv_name = csv_name)
#     # Step 1: Read CSV
#     # df = pd.read_csv(csv_path)
#     # if csv_column not in df.columns:
#     #     raise ValueError(f"Column '{csv_column}' not found in {csv_path}")
#     #
#     # video_paths = df[csv_column].tolist()
#
#     # Step 2: Initialize Preprocessor
#     # preproc = VideoPreprocessor_FANET(
#     #     batch_size=batch_size,
#     #     output_base_dir=output_dir,
#     #     device="cuda"  # auto-handled per rank
#     #     # use_fp16=True
#     #
#     # )
#
#
#     # Step 3: Preprocess all videos
#     # preproc.parallel_main(video_paths)
#
#     print(f"✅ All videos preprocessed and saved to: {output_dir}")
#
#
# def preprocess_videos_before_training(csv_path, csv_column, output_dir, batch_size=128):
#     """
#     Reads video paths from a CSV file and preprocesses them into a common output directory.
#
#     Args:
#         csv_path (str): Path to the CSV file.
#         csv_column (str): Column in CSV containing video paths.
#         output_dir (str): Directory where lip-only videos will be saved.
#         batch_size (int): Number of frames per batch for lip extraction.
#     """
#     import pandas as pd
#     project_dir_curr = get_project_root()
#     csv_name = Path(csv_path).name
#     _,video_paths,_ = create_file_paths(project_dir_curr,csv_name = csv_name)
#     # Step 1: Read CSV
#     # df = pd.read_csv(csv_path)
#     # if csv_column not in df.columns:
#     #     raise ValueError(f"Column '{csv_column}' not found in {csv_path}")
#     #
#     # video_paths = df[csv_column].tolist()
#
#     # # Step 2: Initialize Preprocessor
#     # preproc = VideoPreprocessor_FANET(
#     #     batch_size=batch_size,
#     #     output_base_dir=output_dir,
#     #     device="cuda" # auto-handled per rank
#     #     # use_fp16=True
#     #
#     # )
#     parallel_main(video_paths,batch_size,output_dir)
#     # Step 3: Preprocess all videos
#     # preproc.parallel_main(video_paths)
#
#     print(f"✅ All videos preprocessed and saved to: {output_dir}")
#
#
# from pathlib import Path
# import pandas as pd
#
#
# def create_file_paths(project_dir_curr, csv_name="training_data_two.csv"):
#     """
#     Generates full paths for video files based on filenames from a CSV file,
#     appending '_lips_only' to each filename before the extension.
#
#     Args:
#         project_dir_curr (Path or str): Base project directory.
#         csv_name (str): Name of CSV file with file listings and labels.
#
#     Returns:
#         tuple: lips_only_paths, original_paths, labels
#     """
#     project_dir_curr = Path(project_dir_curr)
#
#     # CSV and video directory paths
#     csv_path = project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / csv_name
#     csv_dir = project_dir_curr / "datasets" / "processed" /  "lav_df" / "train"
#     df = pd.read_csv(csv_path)
#     project_dir_curr = Path("Video-Swin-Transformer")
#     video_dir = project_dir_curr / "data" / "train" / "real"
#
#     # Read CSV and extract file paths
#
#
#     lips_only_paths = []
#     original_paths = []
#     for filename in df['video_file']:
#         original_path = Path(filename)
#         new_filename = original_path.stem + "_lips_only" + original_path.suffix
#         full_lips_only_path = video_dir / new_filename
#         full_original_path = csv_dir / original_path
#
#         lips_only_paths.append(full_lips_only_path)
#         original_paths.append(full_original_path)
#
#     labels = df['label'].tolist()
#
#     return lips_only_paths, original_paths, labels
#
# def create_file_paths_for_train(csv_path, video_dir):
#     """
#     Generates full paths for video files based on filenames from a CSV file.
#
#     Args:
#         csv_path (str or Path): Path to the CSV file containing filenames and labels.
#         video_dir (str or Path): Directory where the corresponding video files are stored
#                                  (from convert_paths_for_training).
#
#     Returns:
#         tuple: (original_paths, labels)
#     """
#     csv_path = Path(csv_path)
#     video_dir = Path(video_dir)
#
#     df = pd.read_csv(csv_path)
#
#     original_paths = []
#     for filename in df['file']:
#         # Build full path inside the video_dir
#         full_original_path = video_dir / filename
#         original_paths.append(str(full_original_path))
#
#     labels = df['label'].tolist()
#
#     return original_paths, labels
#
# import pandas as pd
# from pathlib import Path
#
# def create_file_paths_for_evaluation(fake_csv_path, fake_video_dir, real_csv_path, real_video_dir):
#     """
#     Generates full paths for fake and real video files based on filenames from their CSVs.
#
#     Args:
#         fake_csv_path (str or Path): Path to the CSV file listing fake video filenames + labels.
#         fake_video_dir (str or Path): Directory containing the corresponding fake video files.
#         real_csv_path (str or Path): Path to the CSV file listing real video filenames + labels.
#         real_video_dir (str or Path): Directory containing the corresponding real video files.
#
#     Returns:
#         tuple: (all_paths, labels, fake_paths, real_paths)
#             - all_paths: combined list of fake + real video paths
#             - labels: combined list of labels (from CSVs)
#             - fake_paths: list of fake video paths
#             - real_paths: list of real video paths
#     """
#     fake_csv_path = Path(fake_csv_path)
#     real_csv_path = Path(real_csv_path)
#     fake_video_dir = Path(fake_video_dir)
#     real_video_dir = Path(real_video_dir)
#
#     # --- Fake files ---
#     df_fake = pd.read_csv(fake_csv_path)
#     fake_paths = [(fake_video_dir / fname) for fname in df_fake['file']]
#     fake_labels = df_fake['label'].tolist()
#
#     # --- Real files ---
#     df_real = pd.read_csv(real_csv_path)
#     real_paths = [(real_video_dir / fname) for fname in df_real['file']]
#     real_labels = df_real['label'].tolist()
#
#     # --- Combine ---
#     all_paths = [str(p) for p in (fake_paths + real_paths)]
#     labels = fake_labels + real_labels
#
#     return all_paths, labels, [str(p) for p in fake_paths], [str(p) for p in real_paths]
#
# def create_file_paths_for_inference_ssl(project_dir_curr, csv_name="sampled_combined_data.csv"):
#     """
#     Generates full paths for video files based on filenames from a CSV file
#     for inference evaluation (no lips-only versions).
#
#     Args:
#         project_dir_curr (Path or str): Base project directory.
#         csv_name (str): Name of CSV file with file listings and labels.
#
#     Returns:
#         tuple: original_paths, labels
#     """
#     project_dir_curr = Path(project_dir_curr)
#
#     # CSV and video directory paths
#     csv_path = project_dir_curr / "datasets" / "processed" / "csv_files" / "dfdc" / "inference_data" / csv_name
#     video_dir = project_dir_curr / "datasets" / "processed" / "dfdc" / "eval"
#     df = pd.read_csv(csv_path)
#
#     original_paths = []
#     for filename in df['filename']:
#         full_original_path = video_dir / filename
#         original_paths.append(full_original_path)
#
#     labels = df['label'].tolist()
#
#     return original_paths, labels
#
# def get_project_root(project_name=None):
#     """
#     Locate the root directory of the project based on script path.
#
#     Args:
#         project_name (str, optional): Specific project name to locate.
#
#     Returns:
#         Path or None: Root directory if found, else None.
#     """
#     import os
#     current = Path(os.getcwd()).resolve()
#
#     # Look for the known parent folder
#     for parent in current.parents:
#         if parent.name == "thesis_main_files":
#             base_dir = parent.parent
#             break
#     else:
#         return None
#
#     if project_name:
#         # Return the matching subdirectory if it exists
#         target_path = base_dir / project_name
#         if target_path.exists() and target_path.is_dir():
#             return target_path
#         else:
#             return None
#     else:
#         # Fallback search for common project directories
#         project_names = {"thesis_main_files", "Video-Swin-Transformer", "melodyExtraction_JDC"}
#         for parent in current.parents:
#             if parent.name in project_names:
#                 return parent
#     return None
#
# def convert_paths_for_svm_train_preprocess():
#     """
#     Prepare all necessary paths for SVM training data processing and feature extraction.
#
#     Returns:
#         Tuple containing all path strings used for video preprocessing and feature extraction.
#     """
#     project_dir_curr = get_project_root()
#
#     # Paths for SVM training data
#     csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "final_training_data_svm"/ "training_data_svm_final.csv")
#     video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "checks" / "data_to_preprocess_for_svm")
#
#     # Swin Transformer project-specific paths (unchanged)
#     project_dir_video_swin = get_project_root("Video-Swin-Transformer")
#     video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
#     real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
#     feature_dir_vid = str(project_dir_video_swin)
#
#     return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path
#
#
# def convert_paths_for_svm_val_preprocess():
#     """
#     Prepare all necessary paths for SVM validation data processing and feature extraction.
#
#     Returns:
#         Tuple containing all path strings used for video preprocessing and feature extraction.
#     """
#     project_dir_curr = get_project_root()
#
#     # Paths for SVM validation data
#     csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "val_data_for_svm.csv")
#     video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "checks" / "data_to_preprocess_for_svm_val")
#
#     # Swin Transformer project-specific paths (unchanged)
#     project_dir_video_swin = get_project_root("Video-Swin-Transformer")
#     video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
#     real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
#     feature_dir_vid = str(project_dir_video_swin)
#
#     return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path
# def convert_paths_for_inference_ssl_dfdc():
#     """
#     Prepare all necessary paths for Inference Evaluation preprocessing and feature extraction.
#
#     Returns:
#         Tuple containing all path strings used for video preprocessing and feature extraction.
#     """
#     project_dir_curr = get_project_root()
#
#     # Paths for Inference Evaluation Data
#     csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "dfdc" / "inference_data" / "sampled_combined_data.csv")
#     video_dir = str(project_dir_curr / "datasets" / "processed" / "dfdc" / "eval")  # <-- Assuming videos are here. Adjust if needed.
#
#     # Swin Transformer project-specific paths (unchanged)
#     project_dir_video_swin = get_project_root("Video-Swin-Transformer")
#     video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
#     real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
#     feature_dir_vid = str(project_dir_video_swin)
#
#     return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path
#
# def convert_paths():
#     """
#     Prepare all necessary paths for processing and feature extraction.
#
#     Returns:
#         Tuple containing all path strings used for video preprocessing and feature extraction.
#     """
#     # project_dir_curr = Path("/content/project_combined_repo_clean/thesis_main_files")
#     project_dir_curr = get_project_root()
#
#     # Construct paths used in processing
#     csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "training_data_two.csv")
#     video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "train")
#
#     # Swin Transformer project-specific paths
#     project_dir_video_swin = get_project_root("Video-Swin-Transformer")
#     video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
#     real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
#     feature_dir_vid = str(project_dir_video_swin)
#
#     return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path
# from pathlib import Path
#
# def convert_paths_for_training(csv_name: str = "sample_real_70_percent_half1.csv"):
#     """
#     Prepare all necessary paths for processing and feature extraction (training).
#     CSV file stays with .csv extension, but the video directory uses the filename without extension.
#     """
#     project_dir_curr = get_project_root()
#
#     # Build paths
#     csv_path = str(
#         project_dir_curr / "datasets" / "processed" / "csv_files"
#         / "lav_df" / "new_setup" / "train_files" / csv_name
#     )
#
#     video_dir_name = Path(csv_name).stem  # remove .csv extension
#     video_dir = str(
#         project_dir_curr / "datasets" / "processed" / "lav_df"
#         / "new_setup" / "train_files" / video_dir_name
#     )
#
#     return csv_path, video_dir
#
#
# def convert_paths_for_evaluation(fake_csv_name: str):
#     """
#     Prepare all necessary paths for processing and feature extraction (evaluation).
#     For evaluation, fake and real csvs are mapped to their respective directories.
#     Directories use the filename without extension.
#     """
#     project_dir_curr = get_project_root()
#
#     # --- Derive real_csv_name from fake_csv_name ---
#     base_name = Path(fake_csv_name).stem
#     if "_" not in base_name:
#         raise ValueError(f"Unexpected fake_csv_name format: {fake_csv_name}")
#
#     suffix = base_name.split("_")[-1]  # e.g. "ge7p5" or "lt7p5"
#     real_csv_name = f"{suffix}.csv"
#
#     # --- Construct fake paths ---
#     fake_csv_path = str(
#         project_dir_curr / "datasets" / "processed" / "csv_files"
#         / "lav_df" / "new_setup" / "evaluate_files" / "evaluate" / "fake_files" / fake_csv_name
#     )
#     fake_video_dir_name = Path(fake_csv_name).stem
#     fake_video_dir = str(
#         project_dir_curr / "datasets" / "processed" / "lav_df"
#         / "new_setup" / "evaluate_files" / "evaluate" / "fake_files" / fake_video_dir_name
#     )
#
#     # --- Construct real paths ---
#     real_csv_path = str(
#         project_dir_curr / "datasets" / "processed" / "csv_files"
#         / "lav_df" / "new_setup" / "evaluate_files" / "evaluate" / "real_file_equivalent" / real_csv_name
#     )
#     real_video_dir_name = Path(real_csv_name).stem
#     real_video_dir = str(
#         project_dir_curr / "datasets" / "processed" / "lav_df"
#         / "new_setup" / "evaluate_files" / "evaluate" / "real_file_equivalent" / real_video_dir_name
#     )
#
#     return fake_csv_path, fake_video_dir, real_csv_path, real_video_dir
#
#
# ###############################################################################
# # COMPONENT + FEATURE EXTRACTION CLASSES (Audio restored via video paths)
# ###############################################################################
#
# class VideoComponentExtractor:
#     """
#     Handles raw video component extraction using the FANET video preprocessor.
#     """
#     def extract_video_components(self, video_paths, video_save_dir, output_txt_file, batch_size, video_preprocessor):
#         try:
#             # Process video paths using preprocessor
#             return_paths = video_preprocessor.parallel_main(video_paths)
#             return return_paths.copy()
#         except Exception as e:
#             print(f"Error preprocessing video paths {video_paths}: {e}")
#             return []
# ## NEW fixed GPU code
# class VideoAudioFeatureExtractor:
#     def __init__(self, device=None, amp=True, save_audio_feats=False, audio_save_dir=None):
#         # ✅ DEVICE-ONLY FIX: Proper device initialization
#         if device is None:
#             device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
#         elif isinstance(device, str):
#             device = torch.device(device)
#         # device is now guaranteed to be a torch.device object
#         # END DEVICE FIX
#
#         self.mvit_adapter = MViTVideoFeatureExtractor(
#             device=device,  # Pass the properly initialized device
#             amp=True,
#             strict_temporal=False,
#             save_video_feats=False,
#             save_dir=None,
#             preserve_temporal=True,
#             temporal_pool=True,
#             aggregate="mean"
#         )
#
#         self.audio_extractor = ASTAudioExtractor(
#             device=device,  # Pass the same device
#             amp=amp,
#             time_series=True,
#             token_pool="none",
#             verbose=False,
#             default_save_dir=(audio_save_dir if save_audio_feats else None),
#         )
#
#     # ... [Keep all other methods unchanged] ...
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
#         print(f"🔊 [TRAINING] Starting audio extraction for {len(video_paths)} paths")
#         print(f"🔊 [TRAINING] First 3 paths: {video_paths[:3]}")
#
#         try:
#             items = self.audio_extractor.extract_from_paths(
#                 video_paths,
#                 save=(self.audio_extractor.default_save_dir is not None),
#                 save_dir=self.audio_extractor.default_save_dir,
#                 overwrite=False
#             )
#
#             print(f"🔊 [TRAINING] extract_from_paths returned {len(items)} items")
#
#             # Debug items before feature extraction
#             for i, item in enumerate(items[:3]):
#                 if item is None:
#                     print(f"🔊 [TRAINING] Item {i}: None")
#                 else:
#                     print(
#                         f"🔊 [TRAINING] Item {i}: {type(item)}, keys: {list(item.keys()) if isinstance(item, dict) else 'not dict'}")
#
#             feats = [it["features"] for it in items]  # CPU tensors
#             print(f"🔊 [TRAINING] Feats extracted: {len(feats)}")
#             print(f"🔊 [TRAINING] Feat types: {[type(f) for f in feats[:3]]}")
#
#             shapes = {tuple(f.shape) for f in feats}
#             print(f"🔊 [TRAINING] Length of shapes: {len(shapes)}, Shapes: {shapes}")
#             if len(shapes) == 1:
#                 batch = torch.stack(feats, dim=0)
#             else:
#                 batch = torch.stack([f.mean(0) if f.dim() == 2 else f for f in feats], dim=0)
#             # put on the same device as the extractor (trainer’s rank)
#             return batch.to(self.audio_extractor.device, non_blocking=True)
#         except Exception as e:
#             print(f"[AudioFeat] Error on {len(video_paths)} paths, e.g. {video_paths[:3]}... | {type(e).__name__}: {e}")
#             return None
#
#
# class VideoAudioFeatureProcessor:
#     """
#     MODIFIED: Now supports dedicated GPU allocation for feature extraction
#     """
#
#     def __init__(self, batch_size, video_gpu_id=0, audio_gpu_id=0, verbose=False):
#         self.batch_size = batch_size
#         self.video_gpu_id = video_gpu_id
#         self.audio_gpu_id = audio_gpu_id
#         self.verbose = verbose
#
#         # Create devices for dedicated extraction
#         self.video_device = torch.device(f"cuda:{video_gpu_id}")
#         self.audio_device = torch.device(f"cuda:{audio_gpu_id}")
#
#         if self.verbose:
#             print(f"🎬 Video extraction device: {self.video_device}")
#             print(f"🔊 Audio extraction device: {self.audio_device}")
#
#         # Force feature extractor to use dedicated GPUs
#         self.feature_extractor = VideoAudioFeatureExtractor(
#             device=self.video_device,
#             amp=True,
#             save_audio_feats=False,
#             audio_save_dir=None
#         )
#
#         # Override audio extractor device explicitly
#         self.feature_extractor.audio_extractor.device = self.audio_device
#
#     def create_datasubset(self, csv_path, use_preprocessed=True, video_paths=None,
#                           audio_paths=None, video_save_dir=None, output_txt_file=None):
#         """
#         MODIFIED: Enhanced with GPU switching and error handling
#         """
#         processed_video_features = None
#         processed_audio_features = None
#         video_error = False
#         audio_error = False
#
#         # Temporary GPU context switching for extraction
#         original_device = torch.cuda.current_device()
#
#         try:
#             # Extract video features on dedicated video GPU
#             torch.cuda.set_device(self.video_device)
#             processed_video_features = self.feature_extractor.extract_video_features(video_paths)
#
#             if processed_video_features is not None:
#                 # Move to CPU to avoid cross-GPU transfer issues
#                 processed_video_features = processed_video_features.cpu()
#
#         except Exception as e:
#             print(f"❌ Video feature extraction error on {self.video_device}: {e}")
#             video_error = True
#
#         try:
#             # Extract audio features on dedicated audio GPU
#             torch.cuda.set_device(self.audio_device)
#             processed_audio_features = self.feature_extractor.extract_audio_features(
#                 video_paths, self.batch_size
#             )
#
#             if processed_audio_features is not None:
#                 # Move to CPU to avoid cross-GPU transfer issues
#                 processed_audio_features = processed_audio_features.cpu()
#
#         except Exception as e:
#             print(f"❌ Audio feature extraction error on {self.audio_device}: {e}")
#             audio_error = True
#
#         # Restore original GPU context
#         torch.cuda.set_device(original_device)
#
#         # Return features only if both succeeded
#         if not audio_error and not video_error:
#             return processed_audio_features, processed_video_features
#         else:
#             print("⚠️  Feature extraction errors encountered. Returning None.")
#             return None, None
#
# ######## OLD PRE-GPU STRATEGY CLASS
# # class VideoAudioFeatureProcessor:
# #     """
# #     Combines component and feature extractors to produce a usable dataset.
# #     """
# #     def __init__(self,batch_size):
# #         # self.video_prep          rocess_dir = video_preprocess_dir
# #         # self.feature_dir_vid = feature_dir_vid
# #
# #         # # Initialize the video preprocessor (FANET)
# #         # self.video_preprocessor = VideoPreprocessor_PIPNet(
# #         #     # batch_size=batch_size,
# #         #     output_base_dir_real=video_save_dir,
# #         #     real_output_txt_path=output_txt_file
# #         # )
# #
# #         # Initialize audio preprocessor
# #         # self.audio_preprocessor = AudioPreprocessor()
# #
# #         # Initialize feature extractor (Swin Transformer)
# #         # self.video_feature_ext = VideoAudioFeatureExtractor()
# #         # self.video_feature_ext = mvit_extractor
# #         # self.component_extractor = VideoComponentExtractor()
# #         self.feature_extractor = VideoAudioFeatureExtractor()  # pass rank device
# #         self.batch_size = batch_size
# #
# #     def create_datasubset(self, csv_path, use_preprocessed=True, video_paths=None, audio_paths = None, video_save_dir=None, output_txt_file=None):
# #         processed_video_features = None
# #         processed_audio_features = None
# #         video_error = False
# #         audio_error = False
# #
# #         # try:
# #         #     # Extract components from raw videos
# #         #     preprocessed_video_paths = self.component_extractor.extract_video_components(
# #         #         video_paths, video_save_dir, output_txt_file, self.batch_size, self.video_preprocessor)
# #         # except Exception as e:
# #         #     print(f"Video Component Extraction Error: {e}")
# #         #     video_error = True
# #
# #         try:
# #             # Extract features from video if component extraction succeeded
# #             # if not video_error:
# #             processed_video_features = self.feature_extractor.extract_video_features(video_paths)
# #         except Exception as e:
# #             print(f"Video Feature Extraction Error: {e}")
# #             video_error = True
# #
# #         try:
# #             # Extract audio features using video file paths
# #             processed_audio_features = self.feature_extractor.extract_audio_features(video_paths, self.batch_size)
# #         except Exception as e:
# #             print(f"Audio Feature Extraction Error: {e}")
# #             audio_error = True
# #
# #         # Return features only if both succeeded
# #         if not audio_error and not video_error:
# #             return processed_audio_features, processed_video_features
# #         else:
# #             print("Errors encountered. No features returned.")
# #             return None, None
#
# ###############################################################################
# # DATASET CLASS
# ###############################################################################
#
#
#
# class VideoAudioDataset(Dataset):
#     """
#     Dataset for loading video/audio samples for training.
#     Expects a CSV and a video directory, then builds full file paths.
#     """
#
#     def __init__(self, csv_path: str, video_dir: str, transform=None):
#         """
#         Args:
#             csv_path (str): Path to the CSV file with 'file' and 'label' columns.
#             video_dir (str): Directory containing the corresponding video files.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.csv_path = Path(csv_path)
#         self.video_dir = Path(video_dir)
#         self.transform = transform
#
#         # Use helper to build paths + labels
#         self.video_paths, self.labels = create_file_paths_for_train(self.csv_path, self.video_dir)
#
#         assert len(self.video_paths) == len(self.labels), (
#             f"Mismatch between number of files ({len(self.video_paths)}) "
#             f"and labels ({len(self.labels)}). Check {self.csv_path}"
#         )
#
#     def __len__(self):
#         return len(self.video_paths)
#
#     def __getitem__(self, idx):
#         video_path = self.video_paths[idx]
#         label = self.labels[idx]
#
#         sample = {
#             "video_path": str(video_path),
#             "label": label
#         }
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
#
#
# class VideoAudioDatasetEval(Dataset):
#     """
#     Dataset for loading fake + real video paths and labels for inference-time evaluation.
#
#     Expects explicit CSV paths and corresponding video directories for both fake and real:
#       - fake_csv_path, fake_video_dir
#       - real_csv_path, real_video_dir
#
#     These should come from convert_paths_for_evaluation(fake_csv_name).
#     """
#
#     def __init__(
#         self,
#         fake_csv_path,
#         fake_video_dir,
#         real_csv_path,
#         real_video_dir,
#         augmentations=None,
#     ):
#         """
#         Args:
#             fake_csv_path (str | Path): CSV file containing 'file' and 'label' for fake samples.
#             fake_video_dir (str | Path): Directory with fake video files (folder named after CSV stem).
#             real_csv_path (str | Path): CSV file containing 'file' and 'label' for real samples.
#             real_video_dir (str | Path): Directory with real video files (folder named after CSV stem).
#             augmentations (callable, optional): Optional transform applied to each video path (string-in, string-out).
#         """
#         self.fake_csv_path = str(fake_csv_path)
#         self.fake_video_dir = str(fake_video_dir)
#         self.real_csv_path = str(real_csv_path)
#         self.real_video_dir = str(real_video_dir)
#         self.augmentations = augmentations
#
#         # Build full path lists + labels using your helper
#         (all_paths,
#          labels,
#          fake_paths,
#          real_paths) = create_file_paths_for_evaluation(
#             self.fake_csv_path,
#             self.fake_video_dir,
#             self.real_csv_path,
#             self.real_video_dir
#         )
#
#         # Public attributes (useful for downstream code)
#         self.video_paths = all_paths              # combined fake + real
#         self.labels = labels
#         self.fake_paths = fake_paths              # subset: only fake
#         self.real_paths = real_paths              # subset: only real
#
#         # Backing list of (path, label) tuples
#         self.data = list(zip(self.video_paths, self.labels))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         video_path, label = self.data[idx]
#
#         # Optional augmentations operate on the path string (kept consistent with your prior interface)
#         if self.augmentations:
#             video_path = self.augmentations(video_path)
#
#         return str(video_path), label
#
# # if __name__ == '__main__':
# #     project_root = get_project_root()
# #     file_paths, file_paths_two, labels= create_file_paths(project_root,"training_data_two.csv")
# #     print(str(file_paths[1:20]))
# #     print(str(file_paths_two[1:20]))
# #

#
from torch.utils.data import Dataset
import torch
from torch.utils.tensorboard.summary import video
# from torch.utils.tensorboard import SummaryWriter

# Importing required modules for video/audio preprocessing and feature extraction
# from thesis_main_files.main_files.preprocessing.art_avdf.art.video_preprocessor_SAVE_FILES_MULTI_GPU import  parallel_main
from thesis_main_files.main_files.feature_extraction.new_file_setups.Video_Feature_Extraction.mvitv2_torchvision.mvit_adapter import MViTVideoFeatureExtractor
# from thesis_main_files.main_files.feature_extraction.art_avdf.art.feature_extractor_ART_Video import SWIN_EXECUTOR as VideoFeatureExtractor
# from thesis_main_files.main_files.preprocessing.art_avdf.art.audio_preprocessorart import AudioPreprocessor
# from thesis_main_files.main_files.preprocessing.art_avdf.art.video_preprocessorart_Fanet_gpu import VideoPreprocessor_FANET
# from thesis_main_files.main_files.preprocessing.art_avdf.art.video_preprocessor_SAVE_FILES_MULTI_GPU import parallel_main
from thesis_main_files.main_files.feature_extraction.new_file_setups.Audio_Feature_Extraction.ast_huggingface.extract_audio_features_from_AST import ASTAudioExtractor

from pathlib import Path
import pandas as pd

# from video_preprocessor_fanet_multi_gpu import VideoPreprocessor_FANET
def preprocess_videos_before_evaluation(csv_path, csv_column, output_dir, batch_size=128):
    """
    Reads video paths from a CSV file and preprocesses them into a common output directory.

    Args:
        csv_path (str): Path to the CSV file.
        csv_column (str): Column CSV containing video paths.
        output_dir (str): Directory where lip-only videos will be saved.
        batch_size (int): Number of frames per batch for lip extraction.
    """
    import pandas as pd
    project_dir_curr = get_project_root()
    csv_name = Path(csv_path).name
    _,video_paths,_ = create_file_paths(project_dir_curr,csv_name = csv_name)
    # Step 1: Read CSV
    # df = pd.read_csv(csv_path)
    # if csv_column not in df.columns:
    #     raise ValueError(f"Column '{csv_column}' not found in {csv_path}")
    #
    # video_paths = df[csv_column].tolist()

    # Step 2: Initialize Preprocessor
    # preproc = VideoPreprocessor_FANET(
    #     batch_size=batch_size,
    #     output_base_dir=output_dir,
    #     device="cuda"  # auto-handled per rank
    #     # use_fp16=True
    #
    # )


    # Step 3: Preprocess all videos
    # preproc.parallel_main(video_paths)

    print(f"âœ… All videos preprocessed and saved to: {output_dir}")


def preprocess_videos_before_training(csv_path, csv_column, output_dir, batch_size=128):
    """
    Reads video paths from a CSV file and preprocesses them into a common output directory.

    Args:
        csv_path (str): Path to the CSV file.
        csv_column (str): Column in CSV containing video paths.
        output_dir (str): Directory where lip-only videos will be saved.
        batch_size (int): Number of frames per batch for lip extraction.
    """
    import pandas as pd
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
    # Step 3: Preprocess all videos
    # preproc.parallel_main(video_paths)

    print(f"âœ… All videos preprocessed and saved to: {output_dir}")


from pathlib import Path
import pandas as pd


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
    csv_path = project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / csv_name
    csv_dir = project_dir_curr / "datasets" / "processed" /  "lav_df" / "train"
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

    Returns:
        tuple: (original_paths, labels)
    """
    csv_path = Path(csv_path)
    video_dir = Path(video_dir)

    df = pd.read_csv(csv_path)

    original_paths = []
    for filename in df['file']:
        # Build full path inside the video_dir
        full_original_path = video_dir / filename
        original_paths.append(str(full_original_path))

    labels = df['label'].tolist()

    return original_paths, labels

import pandas as pd
from pathlib import Path

def create_file_paths_for_evaluation(fake_csv_path, fake_video_dir, real_csv_path, real_video_dir):
    """
    Generates full paths for fake and real video files based on filenames from their CSVs.

    Args:
        fake_csv_path (str or Path): Path to the CSV file listing fake video filenames + labels.
        fake_video_dir (str or Path): Directory containing the corresponding fake video files.
        real_csv_path (str or Path): Path to the CSV file listing real video filenames + labels.
        real_video_dir (str or Path): Directory containing the corresponding real video files.

    Returns:
        tuple: (all_paths, labels, fake_paths, real_paths)
            - all_paths: combined list of fake + real video paths
            - labels: combined list of labels (from CSVs)
            - fake_paths: list of fake video paths
            - real_paths: list of real video paths
    """
    fake_csv_path = Path(fake_csv_path)
    real_csv_path = Path(real_csv_path)
    fake_video_dir = Path(fake_video_dir)
    real_video_dir = Path(real_video_dir)

    # --- Fake files ---
    df_fake = pd.read_csv(fake_csv_path)
    fake_paths = [(fake_video_dir / fname) for fname in df_fake['file']]
    fake_labels = df_fake['label'].tolist()

    # --- Real files ---
    df_real = pd.read_csv(real_csv_path)
    real_paths = [(real_video_dir / fname) for fname in df_real['file']]
    real_labels = df_real['label'].tolist()

    # --- Combine ---
    all_paths = [str(p) for p in (fake_paths + real_paths)]
    labels = fake_labels + real_labels

    return all_paths, labels, [str(p) for p in fake_paths], [str(p) for p in real_paths]

def create_file_paths_for_inference_ssl(project_dir_curr, csv_name="sampled_combined_data.csv"):
    """
    Generates full paths for video files based on filenames from a CSV file
    for inference evaluation (no lips-only versions).

    Args:
        project_dir_curr (Path or str): Base project directory.
        csv_name (str): Name of CSV file with file listings and labels.

    Returns:
        tuple: original_paths, labels
    """
    project_dir_curr = Path(project_dir_curr)

    # CSV and video directory paths
    csv_path = project_dir_curr / "datasets" / "processed" / "csv_files" / "dfdc" / "inference_data" / csv_name
    video_dir = project_dir_curr / "datasets" / "processed" / "dfdc" / "eval"
    df = pd.read_csv(csv_path)

    original_paths = []
    for filename in df['filename']:
        full_original_path = video_dir / filename
        original_paths.append(full_original_path)

    labels = df['label'].tolist()

    return original_paths, labels

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

def convert_paths_for_svm_train_preprocess():
    """
    Prepare all necessary paths for SVM training data processing and feature extraction.

    Returns:
        Tuple containing all path strings used for video preprocessing and feature extraction.
    """
    project_dir_curr = get_project_root()

    # Paths for SVM training data
    csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "final_training_data_svm"/ "training_data_svm_final.csv")
    video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "checks" / "data_to_preprocess_for_svm")

    # Swin Transformer project-specific paths (unchanged)
    project_dir_video_swin = get_project_root("Video-Swin-Transformer")
    video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
    real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
    feature_dir_vid = str(project_dir_video_swin)

    return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path


def convert_paths_for_svm_val_preprocess():
    """
    Prepare all necessary paths for SVM validation data processing and feature extraction.

    Returns:
        Tuple containing all path strings used for video preprocessing and feature extraction.
    """
    project_dir_curr = get_project_root()

    # Paths for SVM validation data
    csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "val_data_for_svm.csv")
    video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "checks" / "data_to_preprocess_for_svm_val")

    # Swin Transformer project-specific paths (unchanged)
    project_dir_video_swin = get_project_root("Video-Swin-Transformer")
    video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
    real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
    feature_dir_vid = str(project_dir_video_swin)

    return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path
def convert_paths_for_inference_ssl_dfdc():
    """
    Prepare all necessary paths for Inference Evaluation preprocessing and feature extraction.

    Returns:
        Tuple containing all path strings used for video preprocessing and feature extraction.
    """
    project_dir_curr = get_project_root()

    # Paths for Inference Evaluation Data
    csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "dfdc" / "inference_data" / "sampled_combined_data.csv")
    video_dir = str(project_dir_curr / "datasets" / "processed" / "dfdc" / "eval")  # <-- Assuming videos are here. Adjust if needed.

    # Swin Transformer project-specific paths (unchanged)
    project_dir_video_swin = get_project_root("Video-Swin-Transformer")
    video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
    real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
    feature_dir_vid = str(project_dir_video_swin)

    return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path

def convert_paths():
    """
    Prepare all necessary paths for processing and feature extraction.

    Returns:
        Tuple containing all path strings used for video preprocessing and feature extraction.
    """
    # project_dir_curr = Path("/content/project_combined_repo_clean/thesis_main_files")
    project_dir_curr = get_project_root()

    # Construct paths used in processing
    csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "training_data_two.csv")
    video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "train")

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
        project_dir_curr / "datasets" / "processed" / "csv_files"
        / "lav_df" / "new_setup" / "train_files" / csv_name
    )

    video_dir_name = Path(csv_name).stem  # remove .csv extension
    video_dir = str(
        project_dir_curr / "datasets" / "processed" / "lav_df"
        / "new_setup" / "train_files" / video_dir_name
    )

    return csv_path, video_dir


def convert_paths_for_evaluation(fake_csv_name: str):
    """
    Prepare all necessary paths for processing and feature extraction (evaluation).
    For evaluation, fake and real csvs are mapped to their respective directories.
    Directories use the filename without extension.
    """
    project_dir_curr = get_project_root()

    # --- Derive real_csv_name from fake_csv_name ---
    base_name = Path(fake_csv_name).stem
    if "_" not in base_name:
        raise ValueError(f"Unexpected fake_csv_name format: {fake_csv_name}")

    suffix = base_name.split("_")[-1]  # e.g. "ge7p5" or "lt7p5"
    real_csv_name = f"{suffix}.csv"

    # --- Construct fake paths ---
    fake_csv_path = str(
        project_dir_curr / "datasets" / "processed" / "csv_files"
        / "lav_df" / "new_setup" / "evaluate_files" / "evaluate" / "fake_files" / fake_csv_name
    )
    fake_video_dir_name = Path(fake_csv_name).stem
    fake_video_dir = str(
        project_dir_curr / "datasets" / "processed" / "lav_df"
        / "new_setup" / "evaluate_files" / "evaluate" / "fake_files" / fake_video_dir_name
    )

    # --- Construct real paths ---
    real_csv_path = str(
        project_dir_curr / "datasets" / "processed" / "csv_files"
        / "lav_df" / "new_setup" / "evaluate_files" / "evaluate" / "real_file_equivalent" / real_csv_name
    )
    real_video_dir_name = Path(real_csv_name).stem
    real_video_dir = str(
        project_dir_curr / "datasets" / "processed" / "lav_df"
        / "new_setup" / "evaluate_files" / "evaluate" / "real_file_equivalent" / real_video_dir_name
    )

    return fake_csv_path, fake_video_dir, real_csv_path, real_video_dir


###############################################################################
# COMPONENT + FEATURE EXTRACTION CLASSES (Audio restored via video paths)
###############################################################################

# class VideoComponentExtractor:
#     """
#     Handles raw video component extraction using the FANET video preprocessor.
#     """
#     def extract_video_components(self, video_paths, video_save_dir, output_txt_file, batch_size, video_preprocessor):
#         try:
#             # Process video paths using preprocessor
#             return_paths = video_preprocessor.parallel_main(video_paths)
#             return return_paths.copy()
#         except Exception as e:
#             print(f"Error preprocessing video paths {video_paths}: {e}")
#             return []

class VideoAudioFeatureExtractor:
    """
    Responsible for feature extraction from preprocessed video components and audio waveforms.
    """
    def __init__(self, device=None, amp=True, save_audio_feats=False, audio_save_dir=None):
        self.device = device
        self.mvit_adapter = MViTVideoFeatureExtractor(
            device=self.device,  # torch.device(f"cuda:{local_rank}")
            amp=True,  # uses fp16 autocast in your _forward_model
            strict_temporal=False,  # set True to enforce equal T' within a batch
            save_video_feats=False,  # set True if you want .pt saved per sample
            save_dir=None,  # or a path to store .pt files
            preserve_temporal=True,
            temporal_pool=True,
            aggregate="mean"
        )
        self.audio_extractor = ASTAudioExtractor(
            device=self.device,
            amp=amp,
            time_series=True,     # 'yes' by default
            token_pool="none",    # keep time series, no pooling
            verbose=False,
            default_save_dir=(audio_save_dir if save_audio_feats else None),
        )
    def extract_video_features(self, video_paths):
        try:
            # our MViT adapter exposes .execute(video_paths)
            features = self.mvit_adapter.execute(video_paths)
            return features
        except Exception as e:
            print(f"[VideoFeat] Error on {len(video_paths)} paths, e.g. {video_paths[:3]}... | {type(e).__name__}: {e}")
            return None

    def extract_audio_features(self, video_paths, batch_size,save_path = None):
        try:
            items = self.audio_extractor.extract_from_paths(
                video_paths,
                save=(self.audio_extractor.default_save_dir is not None),
                save_dir=self.audio_extractor.default_save_dir,
                overwrite=False
            )
            feats = [it["features"] for it in items]  # CPU tensors
            shapes = {tuple(f.shape) for f in feats}
            if len(shapes) == 1:
                batch = torch.stack(feats, dim=0)
            else:
                batch = torch.stack([f.mean(0) if f.dim() == 2 else f for f in feats], dim=0)
            # put on the same device as the extractor (trainerâ€™s rank)
            return batch.to(self.audio_extractor.device, non_blocking=True)
        except Exception as e:
            print(f"[AudioFeat] Error on {len(video_paths)} paths, e.g. {video_paths[:3]}... | {type(e).__name__}: {e}")
            return None

class VideoAudioFeatureProcessor:
    """
    Combines component and feature extractors to produce a usable dataset.
    """
    def __init__(self,batch_size,local_rank):
        # self.video_preprocess_dir = video_preprocess_dir
        # self.feature_dir_vid = feature_dir_vid

        # # Initialize the video preprocessor (FANET)
        # self.video_preprocessor = VideoPreprocessor_PIPNet(
        #     # batch_size=batch_size,
        #     output_base_dir_real=video_save_dir,
        #     real_output_txt_path=output_txt_file
        # )

        # Initialize audio preprocessor
        # self.audio_preprocessor = AudioPreprocessor()

        # Initialize feature extractor (Swin Transformer)
        # self.video_feature_ext = VideoAudioFeatureExtractor()
        # self.video_feature_ext = mvit_extractor
        # self.component_extractor = VideoComponentExtractor()
        # torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")
        self.feature_extractor = VideoAudioFeatureExtractor(device = self.device)  # pass rank device
        self.batch_size = batch_size

    def create_datasubset(self, csv_path, use_preprocessed=True, video_paths=None, audio_paths = None, video_save_dir=None, output_txt_file=None):
        processed_video_features = None
        processed_audio_features = None
        video_error = False
        audio_error = False

        # try:
        #     # Extract components from raw videos
        #     preprocessed_video_paths = self.component_extractor.extract_video_components(
        #         video_paths, video_save_dir, output_txt_file, self.batch_size, self.video_preprocessor)
        # except Exception as e:
        #     print(f"Video Component Extraction Error: {e}")
        #     video_error = True

        try:
            # Extract features from video if component extraction succeeded
            # if not video_error:
            processed_video_features = self.feature_extractor.extract_video_features(video_paths)
        except Exception as e:
            print(f"Video Feature Extraction Error: {e}")
            video_error = True

        try:
            # Extract audio features using video file paths
            processed_audio_features = self.feature_extractor.extract_audio_features(video_paths, self.batch_size)
        except Exception as e:
            print(f"Audio Feature Extraction Error: {e}")
            audio_error = True

        # Return features only if both succeeded
        if not audio_error and not video_error:
            return processed_audio_features, processed_video_features
        else:
            print("Errors encountered. No features returned.")
            return None, None

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


class VideoAudioDatasetEval(Dataset):
    """
    Dataset for loading fake + real video paths and labels for inference-time evaluation.

    Expects explicit CSV paths and corresponding video directories for both fake and real:
      - fake_csv_path, fake_video_dir
      - real_csv_path, real_video_dir

    These should come from convert_paths_for_evaluation(fake_csv_name).
    """

    def __init__(
        self,
        fake_csv_path,
        fake_video_dir,
        real_csv_path,
        real_video_dir,
        augmentations=None,
    ):
        """
        Args:
            fake_csv_path (str | Path): CSV file containing 'file' and 'label' for fake samples.
            fake_video_dir (str | Path): Directory with fake video files (folder named after CSV stem).
            real_csv_path (str | Path): CSV file containing 'file' and 'label' for real samples.
            real_video_dir (str | Path): Directory with real video files (folder named after CSV stem).
            augmentations (callable, optional): Optional transform applied to each video path (string-in, string-out).
        """
        self.fake_csv_path = str(fake_csv_path)
        self.fake_video_dir = str(fake_video_dir)
        self.real_csv_path = str(real_csv_path)
        self.real_video_dir = str(real_video_dir)
        self.augmentations = augmentations

        # Build full path lists + labels using your helper
        (all_paths,
         labels,
         fake_paths,
         real_paths) = create_file_paths_for_evaluation(
            self.fake_csv_path,
            self.fake_video_dir,
            self.real_csv_path,
            self.real_video_dir
        )

        # Public attributes (useful for downstream code)
        self.video_paths = all_paths              # combined fake + real
        self.labels = labels
        self.fake_paths = fake_paths              # subset: only fake
        self.real_paths = real_paths              # subset: only real

        # Backing list of (path, label) tuples
        self.data = list(zip(self.video_paths, self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]

        # Optional augmentations operate on the path string (kept consistent with your prior interface)
        if self.augmentations:
            video_path = self.augmentations(video_path)

        return str(video_path), label

# if __name__ == '__main__':
#     project_root = get_project_root()
#     file_paths, file_paths_two, labels= create_file_paths(project_root,"training_data_two.csv")
#     print(str(file_paths[1:20]))
#     print(str(file_paths_two[1:20]))
#