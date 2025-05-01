import torch
from torch.utils.data import Dataset

class AudioVideoPathDataset(Dataset):
    def __init__(self, audio_paths, video_paths, labels):
        assert len(audio_paths) == len(video_paths) == len(labels)
        self.audio_paths = audio_paths
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio_paths[idx], self.video_paths[idx], self.labels[idx]
