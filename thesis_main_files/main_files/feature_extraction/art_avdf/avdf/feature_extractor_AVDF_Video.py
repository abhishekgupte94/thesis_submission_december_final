import subprocess
import pickle
import os
import time
from pathlib import Path
import torch
import sys

class SWIN_EXECUTOR:
    def __init__(self):
        self.CHECKPOINT_PATH = "checkpoints/swin_base_patch244_window877_kinetics600_22k.pth"
        self.CONFIG_PATH = "configs/recognition/swin/custom_swin_feature_extraction.py"
        self.vst_project = self.get_project_root("Video-Swin-Transformer")  # Keep it as a Path object
        self.features_path = str(self.vst_project / "batch_features_videos.pt")  # Convert to string after joining the path
    def get_project_root(self,project_name=None):
        current = Path(__file__).resolve()

        # Locate the parent directory one level above 'thesis_main_files'
        for parent in current.parents:
            if parent.name == "thesis_main_files":
                base_dir = parent.parent  # One level above 'thesis_main_files'
                break
        else:
            return None  # Return None if 'thesis_main_files' is not found in the parent chain

        if project_name:
            # Search specifically for the desired project_name within the base_dir
            target_path = base_dir / project_name
            if target_path.exists() and target_path.is_dir():
                return target_path
            else:
                return None  # Return None if the specified project name is not found
        else:
            # If no project name is specified, search for known projects
            project_names = {"thesis_main_files", "Video-Swin-Transformer"}
            for parent in current.parents:
                if parent.name in project_names:
                    return parent

        return None

    def extract_features(self, return_output=True, max_wait_time=10):
        """Runs test.py via subprocess and reads the output from a file instead of stdout."""
        print("Extracting features for the batch...")

        try:
            cmd = [
                "python", "tools/test.py", self.CONFIG_PATH, self.CHECKPOINT_PATH, "--out", "batch_features_videos.pkl"," --return_output",
                "--return_video"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.vst_project,
                check=True,
                stderr=subprocess.PIPE
            )

            if result.stderr:
                print(f"STDERR Output: {result.stderr.decode()}")  # Debugging stderr

            # Wait for file creation (max_wait_time seconds)
            wait_time = 0

            while not os.path.exists(self.features_path):
                time.sleep(0.5)  # Wait for 500ms
                wait_time += 0.5
                if wait_time >= max_wait_time:
                    print(f"Timeout! {self.features_path} was not created within {max_wait_time} seconds.")
                    return None

            extracted_features = torch.load(self.features_path)

            print(f"\u2705 Feature extraction completed, features shape: {extracted_features.shape}")
            return extracted_features  # Return features directly

        except subprocess.CalledProcessError as e:
            print(f"Feature extraction failed! Error: {e}")
            print(f"STDERR: {e.stderr.decode() if e.stderr else 'No errors'}")
            return None  # Indicate failure

    def execute_swin(self, return_output=True):
        """Runs the feature extraction pipeline, either saving or returning extracted features."""
        print("Feature Extraction Pipeline Starting...")
        if return_output is True:
            features_out = self.extract_features(return_output=return_output)

            if features_out is None:
                print("Skipping batch due to feature extraction failure.")
                return None
            print(f"Successfully received features of length {features_out}")
            return features_out
        return 0  # Return features for direct use

# Example Usage
# sw = SWIN_EXECUTOR()
# features_out = sw.execute_swin()
# print(features_out.shape)
