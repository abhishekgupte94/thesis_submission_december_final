from torch.utils.data import DataLoader, TensorDataset,Dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch

class FeatureBuilder:
    @staticmethod
    def build_dataset(dataset, model_inference_fn, binary_label_fn, batch_size=64, device='mps'):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_features, all_labels = [], []

        with torch.no_grad():
            for audio_feats, video_feats, labels in loader:
                audio_feats = audio_feats.to(device)
                video_feats = video_feats.to(device)

                feat1, feat2 = model_inference_fn(audio_feats, video_feats)
                combined_features = torch.cat([feat1, feat2], dim=1)

                binary_labels = torch.tensor([binary_label_fn(lbl.item()) for lbl in labels], dtype=torch.long)
                all_features.append(combined_features.cpu())
                all_labels.append(binary_labels)

        return TensorDataset(torch.cat(all_features), torch.cat(all_labels))
