from torch.utils.data import Dataset

class AudioVideoInferenceDataset(Dataset):
    def __init__(self, audio_features, video_features, labels):
        assert len(audio_features) == len(video_features) == len(labels)
        self.audio = audio_features
        self.video = video_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio[idx], self.video[idx], self.labels[idx]
