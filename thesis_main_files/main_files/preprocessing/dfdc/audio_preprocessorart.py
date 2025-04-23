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
import subprocess
import gc
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



    # ...

    def extract_audio_from_video(self, video_path, output_dir, sample_rate=16000):
        os.makedirs(output_dir, exist_ok=True)
        output_filename = Path(video_path).stem + ".wav"
        output_path = os.path.join(output_dir, output_filename)

        command = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            output_path
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed for {video_path}:\n{e.stderr.decode()}") from e

        return output_path

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

    def main_processing(self, video_paths, audio_output_dir, batch_size=32, save_path=None):
        audio_paths = []

        # Step 1: Extract audio from each video
        for video_path in video_paths:
            try:
                audio_path = self.extract_audio_from_video(video_path, audio_output_dir, self.sample_rate)
                audio_paths.append(os.path.basename(audio_path))  # only the file name, used by downstream code
                print(f"✅ Extracted: {audio_path}")
            except Exception as e:
                print(f"❌ Failed to extract audio from {video_path}: {e}")

        if not audio_paths:
            print("No audio files were processed.")
            return None

        # Step 2: Run mel + melody processing
        log_mel_list, melody_list = self.process(audio_paths, batch_size=batch_size)

        # Step 3: Combine features
        concatenated_features = self.concatenate_features(log_mel_list, melody_list)

        # Step 4: Save if needed
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(concatenated_features, save_path)
            print(f"📦 Saved concatenated features: {save_path}")
        else:
            print("⚠️ No save_path provided, returning tensor.")
            return concatenated_features

        # Step 5: Clean up memory
        del log_mel_list, melody_list, concatenated_features
        gc.collect()
        torch.cuda.empty_cache()

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