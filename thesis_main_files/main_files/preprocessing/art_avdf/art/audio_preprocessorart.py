import os
import numpy as np
import torch
import torchaudio
import pandas as pd
import gc
from pathlib import Path
import psutil
import torch.nn.functional as F
# from models.art_avdf.training_pipeline.training_ART import audio_paths

os.environ['KERAS_BACKEND'] = "tensorflow"

from melodyExtraction_JDC.custom.jdc_implementation_for_art_avdf import JDCModel


class AudioPreprocessor:
    def __init__(self, sample_rate=16000, batch_size=32):
        self.project_dir = self.get_project_root()
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=20, n_mels=640, window_fn=torch.hann_window
        )
        self.batch_size = batch_size
        self.jdc_model = JDCModel()

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024

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
            project_names = {"thesis_main_files", "Video-Swin-Transformer","melodyExtraction_JDC"}
            for parent in current.parents:
                if parent.name in project_names:
                    return parent

        return None

    def process_batch(self, audio_paths):
        for audio_path in audio_paths:
            full_audio_path = os.path.join(self.project_dir, "datasets", "processed", "lav_df", "audio_wav", "train", audio_path)
            log_mel_spec = self.process_audio(full_audio_path)
            yield log_mel_spec

    def get_melody_batch(self, audio_paths):
        return self.jdc_model.batch_process_melody(audio_paths, output_dir=None, is_fake=False,
                                                   batch_size=self.batch_size)

    def process_and_save(self, audio_paths, mel_output_dir, melody_output_dir, batch_size=32):
        os.makedirs(mel_output_dir, exist_ok=True)
        os.makedirs(melody_output_dir, exist_ok=True)

        for i in range(0, len(audio_paths), batch_size):
            print(f"Memory usage before processing batch {i // batch_size + 1}: {self.get_memory_usage():.2f} MB")
            batch_audio_paths = audio_paths[i:i + batch_size]

            for idx, log_mel_spec in enumerate(self.process_batch(batch_audio_paths)):
                filename = Path(batch_audio_paths[idx]).stem
                mel_path = os.path.join(mel_output_dir, f"{filename}.pt")
                torch.save(log_mel_spec, mel_path)
                print(f"✅ Saved mel spectrogram: {mel_path}")

            full_audio_paths = [
                os.path.join(self.project_dir, "datasets", "processed", "lav_df", "audio_wav", "train", audio_path) for audio_path in batch_audio_paths
            ]
            melody_features = self.get_melody_batch(full_audio_paths)
            for idx, melody in enumerate(melody_features):
                filename = Path(batch_audio_paths[idx]).stem
                melody_path = os.path.join(melody_output_dir, f"{filename}.pt")
                torch.save(melody, melody_path)
                print(f"✅ Saved melody: {melody_path}")

            print(f"Memory usage after processing batch {i // batch_size + 1}: {self.get_memory_usage():.2f} MB")

    def get_melody_single(self, audio_path, batch_size=1):
        return self.jdc_model.predict(audio_path, output_dir=None, is_fake=False,
                                      batch_size=self.batch_size)

    def process_audio(self, audio_path):
        with torch.no_grad():
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            mel_spec = self.mel_transform(waveform)
            log_mel_spec = torch.log(mel_spec + 1e-9)
        return log_mel_spec



    def pad_or_truncate(self,batch_tensor, target_len, dim=2):
        current_len = batch_tensor.shape[dim]
        if current_len > target_len:
            return batch_tensor.narrow(dim, 0, target_len)  # Truncate
        elif current_len < target_len:
            pad_size = [0] * (2 * batch_tensor.dim())
            pad_size[-(2 * dim + 1)] = target_len - current_len  # Pad at end
            return F.pad(batch_tensor, pad_size)
        return batch_tensor

    # def concatenate_features(self, log_mel_list, melody_list):
    #     """
    #     Process a list of log-mel spectrograms and a list of melody tensors.
    #     For each pair, it trims them to a common time dimension, repeats the melody
    #     along the channel dimension to match log_mel, concatenates along the time axis,
    #     and returns a list of concatenated features.
    #     """
    #     concatenated_list = []
    #     concatenated_list_tensor = None
    #     if len(log_mel_list) != len(melody_list):
    #         raise ValueError("The number of log_mel tensors and melody tensors must match.")
    #
    #     for log_mel, melody in zip(log_mel_list, melody_list):
    #         # Validate tensor shapes
    #         if log_mel.ndim != 3 or melody.ndim != 2:
    #             raise ValueError(f"Unexpected tensor dimensions in {log_mel} or {melody}")
    #
    #         # Ensure both have a common time length
    #         min_T = min(log_mel.shape[2], melody.shape[1])
    #         log_mel = log_mel[:, :, :min_T]
    #         melody = melody[:, :min_T]
    #
    #         # Repeat melody along the channel dimension to match log_mel's channel count
    #         # (Assuming log_mel.shape[1] is the desired number of channels)
    #         repeated_melody = melody.unsqueeze(1).repeat(1, log_mel.shape[1], 1)
    #
    #         # Concatenate along the time dimension (dim=2)
    #         concatenated = torch.cat((log_mel, repeated_melody), dim=2)
    #         concatenated_list.append(concatenated)
    #         concatenated_list_tensor = torch.tensor(concatenated_list)
    #     return concatenated_list_tensor
    def concatenate_features(self, log_mel_list, melody_list):
        concatenated_list = []

        if len(log_mel_list) != len(melody_list):
            raise ValueError("The number of log_mel tensors and melody tensors must match.")

        # Step 1: Find the max length across all tensors
        all_lengths = [log_mel.shape[2] for log_mel in log_mel_list] + \
                      [melody.shape[1] for melody in melody_list]
        max_T = max(all_lengths)

        # Step 2: Pad/Truncate and Concatenate
        for log_mel, melody in zip(log_mel_list, melody_list):
            if log_mel.ndim != 3 or melody.ndim != 2:
                raise ValueError(f"Unexpected tensor dimensions: log_mel={log_mel.shape}, melody={melody.shape}")

            log_mel = self.pad_or_truncate(log_mel, max_T, dim=2)  # shape: (1, 64, max_T)
            melody = self.pad_or_truncate(melody, max_T, dim=1)  # shape: (64, max_T)

            repeated_melody = melody.unsqueeze(0).repeat(log_mel.shape[0], 1, 1)  # shape: (1, 64, max_T)
            concatenated = torch.cat((log_mel, repeated_melody), dim=1)  # shape: (1, 128, max_T)

            concatenated_list.append(concatenated)

        # Step 3: Stack into batch
        batch_tensor = torch.stack(concatenated_list)  # shape: (B, 128, max_T)
        return batch_tensor

    def process(self, audio_paths, batch_size=32):
        all_log_mel_specs = []
        all_melody_features = []

        for i in range(0, len(audio_paths), batch_size):
            print(f"Memory usage before processing batch {i // batch_size + 1}: {self.get_memory_usage():.2f} MB")
            batch_audio_paths = audio_paths[i:i + batch_size]

            for log_mel_spec in self.process_batch(batch_audio_paths):
                all_log_mel_specs.append(log_mel_spec)

            full_audio_paths = [
                os.path.join(self.project_dir, "datasets", "processed", "lav_df", "audio_wav", "train", audio_path)
                for audio_path in batch_audio_paths
            ]
            melody_features = self.get_melody_batch(full_audio_paths)
            all_melody_features.extend(melody_features)

            print(f"Memory usage after processing batch {i // batch_size + 1}: {self.get_memory_usage():.2f} MB")

        return all_log_mel_specs, all_melody_features

    def main_processing(self, audio_paths, batch_size=32,save_path = None):
        log_mel_list, melody_list = self.process(audio_paths, batch_size=batch_size)
        # print(len(melody_list))
        concatenated_features = self.concatenate_features(log_mel_list, melody_list)
        # stacked_tensor = torch.stack(concatenated_features)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(concatenated_features, save_path)
        else:
            return concatenated_features
        return 0


# def get_project_root(project_name=None):
#     current = Path(__file__).resolve()
#
#     # Locate the parent directory one level above 'thesis_main_files'
#     for parent in current.parents:
#         if parent.name == "thesis_main_files":
#             base_dir = parent.parent  # One level above 'thesis_main_files'
#             break
#     else:
#         return None  # Return None if 'thesis_main_files' is not found in the parent chain
#
#     if project_name:
#         # Search specifically for the desired project_name within the base_dir
#         target_path = base_dir / project_name
#         if target_path.exists() and target_path.is_dir():
#             return target_path
#         else:
#             return None  # Return None if the specified project name is not found
#     else:
#         # If no project name is specified, search for known projects
#         project_names = {"thesis_main_files", "Video-Swin-Transformer","melodyExtraction_JDC"}
#         for parent in current.parents:
#             if parent.name in project_names:
#                 return parent
#
#     return None

# def convert_paths():
#     project_dir_curr = get_project_root()
#     csv_path = str(
#         project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "training_data_two.csv")
#     # Audio preprocess path
#     audio_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "audio_wav" / "train")
#     video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" /  "train")
#     audio_preprocess_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "features" / "audio" / "real")
#     feature_dir_audio = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "features" / "audio" / "real")
#     # Video preprocess path
#     project_dir_video_swin = get_project_root("Video-Swin-Transformer")
#     video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
#     real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" /"lip_train_text_real.txt")
#
#     feature_dir_vid = str(project_dir_video_swin)
#     return csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin, video_preprocess_dir, feature_dir_vid, audio_dir,video_dir,real_output_txt_path
#
# def create_dataset(csv_path, num_rows=None, video_dir=None, audio_dir=None):
#     df = pd.read_csv(csv_path)
#     required_cols = {'video_file', 'audio_file', 'label'}
#     if not required_cols.issubset(df.columns):
#         raise ValueError("CSV file must contain 'video_file', 'audio_file', and 'label' columns")
#
#     if num_rows:
#         df = df.iloc[:num_rows]
#
#     video_paths = [str(Path(video_dir) / filename) for filename in df['video_file']]
#     audio_paths = [str(Path(audio_dir) / filename) for filename in df['audio_file']]
#     labels = df['label'].tolist()
#
#     return audio_paths, video_paths, labels
#
# def create_dataset_idx(csv_path, start_idx=0, num_rows=3, video_dir=None, audio_dir=None):
#     df = pd.read_csv(csv_path)
#     required_cols = {'video_file', 'audio_file', 'label'}
#     if not required_cols.issubset(df.columns):
#         raise ValueError("CSV file must contain 'video_file', 'audio_file', and 'label' columns")
#
#     # Slice the DataFrame using the provided start index and number of rows
#     df = df.iloc[start_idx : start_idx + num_rows]
#
#     video_paths = [str(Path(video_dir) / filename) for filename in df['video_file']]
#     audio_paths = [str(Path(audio_dir) / filename) for filename in df['audio_file']]
#     labels = df['label'].tolist()
#
#     return audio_paths, video_paths, labels
#
#
#
#
# # # # print(audio_paths)
# csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin, video_preprocess_dir, feature_dir_vid, audio_dir,video_dir,real_output_txt_path = convert_paths()
# audio_paths, video_paths, labels = create_dataset_idx(csv_path,start_idx = 32, num_rows = 32, video_dir = video_dir, audio_dir = audio_dir)
# ap = AudioPreprocessor()
# concatenated = ap.main_processing(audio_paths,batch_size = 32,save_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_2_audio_embeddings/test_2_audio_embeddings.pt")
# # # print(concatenated)
# # # print(audio_paths)