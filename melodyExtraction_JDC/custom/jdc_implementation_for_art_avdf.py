import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import numpy as np
import torch
import torchaudio
import pandas as pd
import gc
import librosa
from scipy.signal import medfilt
from tensorflow.keras.models import load_model
from melodyExtraction_JDC.model import melody_ResNet_joint_add
from pathlib import Path

os.environ['KERAS_BACKEND'] = "tensorflow"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

class JDCModel:
    def __init__(self):
        self.project_dir = self.get_project_root()
        self.x_mean = str(self.project_dir / "x_data_mean_total_31.npy")
        self.x_std = str(self.project_dir /  "x_data_std_total_31.npy")
        self.model_path = str(self.project_dir /  "weights" / "ResNet_joint_add_L(CE_G).hdf5")
        # Updated Device Selection Logic
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use CUDA GPU
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Use Apple Silicon GPU (MPS)
        else:
            self.device = torch.device("cpu")  # Fallback to CPU

        self.model = self.load_jdc_model()
        self.pitch_range = self.setup_pitch_range()

        self.x_train_mean, self.x_train_std = self.load_normalization_values()

    def extract_spectrogram_from_waveform(self, waveform, sample_rate=16000):
        """
        Extracts spectrogram from in-memory waveform using librosa.
        """
        # Convert PyTorch tensor to numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().cpu().numpy()

        # Compute STFT
        S = librosa.core.stft(waveform, n_fft=1024, hop_length=20, win_length=1024)
        x_spec = np.abs(S)
        x_spec = librosa.core.power_to_db(x_spec, ref=np.max).astype(np.float32)

        # Padding for input size
        num_frames = x_spec.shape[1]
        padNum = num_frames % self.get_options().input_size
        if padNum != 0:
            len_pad = self.get_options().input_size - padNum
            padding_feature = np.zeros(shape=(513, len_pad))
            x_spec = np.concatenate((x_spec, padding_feature), axis=1)

        # Frame slicing
        x_test = np.array([x_spec[:, range(j, j + self.get_options().input_size)].T for j in
                           range(0, num_frames, self.get_options().input_size)])
        x_test = (x_test - self.x_train_mean) / (self.x_train_std + 1e-6)
        x_test = x_test[:, :, :, np.newaxis]

        return x_test

    def predict_from_waveform(self, waveform, sample_rate=16000, batch_size=32):
        """
        Predict melody from in-memory waveform instead of file.
        """
        x_test = self.extract_spectrogram_from_waveform(waveform, sample_rate)
        y_predict = self.model.predict(x_test, batch_size=batch_size, verbose=0)

        # Process prediction
        num_total = y_predict[0].shape[0] * y_predict[0].shape[1]
        est_pitch = np.zeros(num_total)
        y_predict = np.reshape(y_predict[0], (num_total, y_predict[0].shape[2]))

        for i in range(y_predict.shape[0]):
            index_predict = np.argmax(y_predict[i, :])
            pitch_MIDI = self.pitch_range[int(index_predict)]
            if 38 <= pitch_MIDI <= 83:
                est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.) * 440

        est_pitch = medfilt(est_pitch, 5)

        return torch.from_numpy(est_pitch).float().unsqueeze(0)

    def load_jdc_model(self):
        options = self.get_options()
        print(self.project_dir)
        model = melody_ResNet_joint_add(options)
        model.load_weights(self.model_path)
        model.trainable = False
        # model.eval()
        return model

    def get_options(self):
        class Options:
            def __init__(self):
                self.num_spec = 513
                self.input_size = 31
                self.batch_size = 64
                self.resolution = 16
                self.figureON = False

        return Options()

    def setup_pitch_range(self):
        options = self.get_options()
        pitch_range = np.arange(38, 83 + 1.0 / options.resolution, 1.0 / options.resolution)
        return np.concatenate([np.zeros(1), pitch_range])

    def load_normalization_values(self):
        x_train_mean = np.load(self.x_mean)
        x_train_std = np.load(self.x_std)
        return x_train_mean, x_train_std

    def extract_spectrogram(self, file_name):
        y, sr = librosa.load(file_name, sr=16000)
        S = librosa.core.stft(y, n_fft=1024, hop_length=20, win_length=1024)
        x_spec = np.abs(S)
        x_spec = librosa.core.power_to_db(x_spec, ref=np.max).astype(np.float32)
        num_frames = x_spec.shape[1]
        padNum = num_frames % self.get_options().input_size
        if padNum != 0:
            len_pad = self.get_options().input_size - padNum
            padding_feature = np.zeros(shape=(513, len_pad))
            x_spec = np.concatenate((x_spec, padding_feature), axis=1)
        x_test = np.array([x_spec[:, range(j, j + self.get_options().input_size)].T for j in
                           range(0, num_frames, self.get_options().input_size)])
        x_test = (x_test - self.x_train_mean) / (self.x_train_std + 1e-6)
        x_test = x_test[:, :, :, np.newaxis]
        del x_spec, y, S
        gc.collect()
        return x_test

    def predict(self, file_name,batch_size):
        x_test = self.extract_spectrogram(file_name)
        y_predict = self.model.predict(x_test, batch_size=batch_size, verbose=0)
        del x_test
        gc.collect()
        num_total = y_predict[0].shape[0] * y_predict[0].shape[1]
        est_pitch = np.zeros(num_total)
        y_predict = np.reshape(y_predict[0], (num_total, y_predict[0].shape[2]))
        for i in range(y_predict.shape[0]):
            index_predict = np.argmax(y_predict[i, :])
            pitch_MIDI = self.pitch_range[int(index_predict)]
            if 38 <= pitch_MIDI <= 83:
                est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.) * 440
        est_pitch = medfilt(est_pitch, 5)
        del y_predict
        gc.collect()
        return torch.from_numpy(est_pitch).float().unsqueeze(0)

    def batch_process_melody(self, file_names, output_dir, is_fake,batch_size):
        batch_predictions = [self.predict(file,batch_size) for file in file_names]
        # self.batch_save_melody(file_names, batch_predictions, output_dir, is_fake)
        return batch_predictions

    def batch_save_melody(self, file_paths, melodies, output_dir, is_fake):
        sub_dir = "fake" if is_fake else "real"
        save_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(save_dir, exist_ok=True)
        [np.save(os.path.join(save_dir, f"{os.path.splitext(os.path.basename(file))[0]}.npy"), melody.numpy()) for file, melody in zip(file_paths, melodies)]
    def get_project_root(self):
        """Find the project root dynamically based on 'thesis_main_files'.
        If not found, classify as an attached project.
        """
        current = Path(__file__).resolve()

        # Traverse up to find 'thesis_main_files'
        # for parent in current.parents:
        #     if parent.name == "thesis_main_files":
        #         return parent.parents[4]  # Equivalent to .parents[5] in the script

        # If 'thesis_main_files' is not found, classify as attached project
        return current.parents[1]  # Equivalent to .parents[2] in the script

    def batch_melody_waveform(self, waveform_list, batch_size=32):
        """
        Process a list of in-memory waveforms and return melody predictions.

        Args:
            waveform_list (List[Tensor]): List of 1D or 2D torch tensors representing audio waveforms.
            batch_size (int): Batch size for inference.

        Returns:
            List[Tensor]: Predicted melody tensors.
        """
        batch_predictions = []
        for waveform in waveform_list:
            try:
                # Predict melody directly from waveform
                melody = self.predict_from_waveform(waveform, sample_rate=16000, batch_size=batch_size)
                # print(melody.size())
                batch_predictions.append(melody)
            except Exception as e:
                print(f"❌ Error processing waveform: {e}")
                batch_predictions.append(None)  # Or use torch.zeros_like if shape is critical
        return batch_predictions

# print("JDC Model loaded successfully and ready for use.")

