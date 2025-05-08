import subprocess
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
class SWIN_EXECUTOR:
    def __init__(self,video_preprocess_dir = None):
        self.CHECKPOINT_PATH = "checkpoints/swin_base_patch244_window877_kinetics600_22k.pth"
        self.CONFIG_PATH = "configs/recognition/swin/custom_swin_feature_extraction.py"
        self.vst_project = self.get_project_root("Video-Swin-Transformer")
        self.features_path = str(self.vst_project / "batch_features_lips.pt")
        self.video_preprocess_dir = video_preprocess_dir
    def get_project_root(self, project_name=None):
        import os
        current = Path(os.getcwd()).resolve()
        for parent in current.parents:
            if parent.name == "thesis_main_files":
                base_dir = parent.parent
                break
        else:
            return None

        if project_name:
            target_path = base_dir / project_name
            return target_path if target_path.exists() else None

        return None

    def extract_features(self, return_output=True, max_wait_time=10):
        print("Extracting features for the batch...")

        try:
            cmd = [
                "python", "tools/test.py",
                self.CONFIG_PATH,
                self.CHECKPOINT_PATH,
                "return_output",
                "--out", "batch_features_lips.pkl"
                # "--return_video"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.vst_project,
                check=True,
                stderr=subprocess.PIPE
            )

            if result.stderr:
                print(f"STDERR Output: {result.stderr.decode()}")

            wait_time = 0
            time.sleep(10)
            # while not os.path.exists(self.features_path):
            #     time.sleep(0.5)
            #     wait_time += 0.5
            #     if wait_time >= max_wait_time:
            #         print(f"Timeout! {self.features_path} not created in {max_wait_time}s.")
            #         return None

            extracted_features = torch.load(self.features_path)
            # Step 1: Rearrange to (B*T, C, H, W)
            # B, T, H, W, C = extracted_features.shape
            # extracted_features = extracted_features.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, C, H, W)
            # extracted_features = extracted_features.view(B * T, C, H, W)  # (B*T, C, H, W)
            #
            # # Step 2: Apply spatial pooling → (B*T, C, 1, 1)
            # pool = nn.AdaptiveAvgPool2d((1, 1))
            # extracted_features = pool(extracted_features)  # (B*T, C, 1, 1)
            #
            # # Step 3: Flatten → (B*T, C)
            # extracted_features = extracted_features.view(B, T, C)  # (B, T, C)

            # ✅ Now: x is of shape (B, T, D)
            print(extracted_features.shape)  # e.g., torch.Size([2, 16, 1024])

            if extracted_features.numel() == 0:
                print("⚠️ Loaded feature tensor is empty (zero size)! Check data or model output.")
            else:
                print(f"✅ Loaded features shape: {extracted_features.shape}")

            os.remove(self.features_path)
            print(f"Deleted feature file: {self.features_path}")
            # Delete all .mp4 files in video_preprocess_dir
            # if self.video_preprocess_dir:
            #     mp4_files = Path(self.video_preprocess_dir).glob("*.mp4")
            #     for mp4_file in mp4_files:
            #         try:
            #             mp4_file.unlink()
            #             print(f"Deleted: {mp4_file}")
            #         except Exception as e:
            #             print(f"Failed to delete {mp4_file}: {e}")
            return extracted_features

        except subprocess.CalledProcessError as e:
            print(f"Feature extraction subprocess failed! Error: {e}")
            print(f"STDERR: {e.stderr.decode() if e.stderr else 'No errors'}")
            return None

        except Exception as e:
            print(f"Unexpected error during feature extraction: {e}")
            return None

    def execute_swin(self, return_output=True):
        print("Feature Extraction Pipeline Starting...")
        if return_output:
            with torch.no_grad():
                features_out = self.extract_features(return_output=return_output)
                if features_out is None:
                    print("Skipping batch due to feature extraction failure.")
                    return None

            print(f"Successfully received features of shape {features_out.shape}")
            return features_out

        return 0
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
    csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "training_data_two.csv")
    video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "train")

    # Swin Transformer project-specific paths
    project_dir_video_swin = get_project_root("Video-Swin-Transformer")
    video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
    real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
    feature_dir_vid = str(project_dir_video_swin)

    return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path


# from thesis_main_files.models.data_loaders.data_loader_ART import convert_paths,get_project_root
import time
st = time.time()
csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path = convert_paths()
sw = SWIN_EXECUTOR()

extracted_features = sw.execute_swin()
en = time.time()
print(extracted_features)

print(f"total time is equal to {en-st}")