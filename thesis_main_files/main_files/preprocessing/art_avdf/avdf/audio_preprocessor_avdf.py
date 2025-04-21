import torch
import torchaudio
import numpy as np


class AudioPreprocessorAVDF:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,  # 25ms window
            hop_length=640,  # 40ms intervals for alignment with video
            n_mels=64,
            window_fn=torch.hann_window
        )

    def process_audio(self, audio_path):
        # Load and resample audio if necessary
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale and normalize to [0,1] range
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())

        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(0)

        return mel_spec  # Shape: [1, Ca, Ta] where Ca=64 mel channels



