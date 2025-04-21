import os
import torch
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
import psutil
import traceback


class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, batch_size=32):
        self.sample_rate = sample_rate
        self.batch_size = batch_size

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB

    def save_batch_features(self, file_paths, features, output_dir):
        """Save features with error handling"""
        os.makedirs(output_dir, exist_ok=True)
        error_count = 0

        for file, feature in zip(file_paths, features):
            try:
                filename = os.path.splitext(file)[0]
                save_path = os.path.join(output_dir, f"{filename}.pt")

                # Create contiguous copy and save
                feature = feature.contiguous().clone()
                torch.save(feature, save_path)
                print(f"✅ Saved: {save_path}")

            except Exception as e:
                error_count += 1
                print(f"⚠️ Error saving {file}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(traceback.format_exc())
                continue

        print(f"\nSave complete. Success: {len(file_paths) - error_count}, Errors: {error_count}")

    def load_and_concatenate_features(self, log_mel :torch.tensor, melody:torch.tensor):

        # Validate tensor shapes
        if log_mel.ndim != 3 or melody.ndim != 2:
            raise ValueError(f"Unexpected tensor dimensions in {log_mel} or {melody}")

        min_T = min(log_mel.shape[2], melody.shape[1])
        log_mel = log_mel[:, :, :min_T]
        melody = melody[:, :min_T]

        concatenated = torch.cat((log_mel, melody.unsqueeze(1).repeat(1, 64, 1)), dim=2)
        return concatenated

