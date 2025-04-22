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
import torchaudio
import torch
import io
import ffmpeg
os.environ['KERAS_BACKEND'] = "tensorflow"

from melodyExtraction_JDC.custom.jdc_implementation_for_art_avdf import JDCModel


class AudioPreprocessor:
    def __init__(self, sample_rate=16000, batch_size=32):
        self.project_dir = self.get_project_root()
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,  # FFT window size (can be 1024 if you want more frequency resolution)
            hop_length=640,  # Shift by 40ms → 1 frame per 40ms
            win_length=640,  # Match window size to hop length (40ms)
            n_mels=64,
            window_fn=torch.hann_window
        )

        self.batch_size = batch_size
        self.jdc_model = JDCModel()

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024

    def process_audio_from_waveform(self, waveform, sample_rate = 16000):
        """
        Converts waveform to log-mel spectrogram using class-defined mel_transform.
        Handles resampling and mono conversion.
        """
        # print("processing waveform reached")
        with torch.no_grad():
            # Resample if needed
            # if sample_rate != self.sample_rate:
            #     waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

            # Convert to mono if needed
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Apply mel transformation
            mel_spec = self.mel_transform(waveform)

            # Use log scale with numerical stability
            log_mel_spec = torch.log(mel_spec + 1e-9)

        return log_mel_spec

    def process_waveforms(self, video_paths, batch_size=32):
        all_log_mel_specs = []
        all_melody_features = []

        for i in range(0, len(video_paths), batch_size):
            print(f"Memory usage before processing batch {i // batch_size + 1}: {self.get_memory_usage():.2f} MB")
            batch_video_paths = video_paths[i:i + batch_size]

            waveforms_and_sr = [self.extract_audio_tensor_from_video(v_path) for v_path in batch_video_paths]

            for waveform, sr in waveforms_and_sr:
                log_mel_spec = self.process_audio_from_waveform(waveform, sr)
                print(log_mel_spec.size())
                all_log_mel_specs.append(log_mel_spec)

            waveforms = [wf for wf, _ in waveforms_and_sr]
            melody_features = self.get_melody_batch_from_waveforms(waveforms)
            # print(melody_features)
            all_melody_features.extend(melody_features)
            # print("log_mel shape:", all_log_mel_specs.shape)  # Should be [1, 64, T]
            # print("melody shape:", all_melody_features.shape)  # Maybe it's [65, T]

            print(f"Memory usage after processing batch {i // batch_size + 1}: {self.get_memory_usage():.2f} MB")

        return all_log_mel_specs, all_melody_features

    def main_processing_waveforms(self, video_paths, batch_size=32, save_path=None):
        log_mel_list, melody_list = self.process_waveforms(video_paths, batch_size=batch_size)
        concatenated_features = self.align_log_mel_by_melody(log_mel_list, melody_list)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(concatenated_features, save_path)
        else:
            return concatenated_features
        return 0

    def get_melody_batch_from_waveforms(self, waveforms):
        return self.jdc_model.batch_melody_waveform(waveforms,
                                                      batch_size=self.batch_size)

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

    def extract_audio_tensor_from_video(self,video_path, sample_rate=16000):
        command = [
            'ffmpeg',
            '-i', video_path,
            '-f', 'wav',  # output format
            '-acodec', 'pcm_s16le',  # raw PCM audio
            '-ar', str(sample_rate),  # resample to desired rate
            '-ac', '1',  # mono
            'pipe:1'  # output to stdout
        ]
        # Run ffmpeg and capture stdout
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {process.stderr.decode()}")

        # Load the audio from the byte stream
        audio_bytes = io.BytesIO(process.stdout)
        waveform, sr = torchaudio.load(audio_bytes)

        return waveform, sr  # <-- ✅ Return both

    import torch
    import torch.nn.functional as F

    def pad_or_truncate(self,tensor, target_len, dim=2):
        current_len = tensor.shape[dim]
        if current_len > target_len:
            return tensor.narrow(dim, 0, target_len)
        elif current_len < target_len:
            pad_size = [0] * (2 * tensor.dim())
            pad_size[-(2 * dim + 1)] = target_len - current_len  # Pad at end
            return F.pad(tensor, pad_size)
        return tensor

    def align_log_mel_by_melody(self,log_mel_list, melody_list):
        if len(log_mel_list) != len(melody_list):
            raise ValueError("log_mel_list and melody_list must have the same length.")

        # STEP 1: Get the global max T
        max_T = 0
        for log_mel, melody in zip(log_mel_list, melody_list):
            if log_mel.ndim != 3 or log_mel.shape[0] != 1 or log_mel.shape[1] != 64:
                raise ValueError(f"Expected log_mel shape [1, 64, T], got {log_mel.shape}")

            log_mel_T = log_mel.shape[2]
            melody_T = melody.shape[0] if melody.ndim == 1 else melody.shape[1]
            max_T = max(max_T, log_mel_T, melody_T)

        # STEP 2: Pad all log_mel tensors to max_T
        aligned_list = []
        for log_mel in log_mel_list:
            padded = self.pad_or_truncate(log_mel, max_T, dim=2)  # [1, 64, max_T]
            aligned_list.append(padded.unsqueeze(0))  # [1, 1, 64, max_T]

        return torch.cat(aligned_list, dim=0)  # Final shape: [B, 1, 64, max_T]

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
        concatenated_features = self.align_log_mel_by_melody(log_mel_list, melody_list)
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
#
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

    # return audio_paths, video_paths, labels



#
# # # # # print(audio_paths)
# csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin, video_preprocess_dir =convert_paths()
# audio_paths, video_paths, labels = create_dataset_idx(csv_path,start_idx = 32, num_rows = 32, video_dir = video_dir, audio_dir = audio_dir)
# ap = AudioPreprocessor()
# concatenated = ap.main_processing(audio_paths,batch_size = 32,save_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_2_audio_embeddings/test_2_audio_embeddings.pt")
# # # print(concatenated)
# # # print(audio_paths)