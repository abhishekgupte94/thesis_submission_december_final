# Transform lip features via non-linear transformation g(·)
import torch
import torch.nn as nn
import torch.nn.functional as F
from thesis_main_files.config import CONFIG
from thesis_main_files.misc_tests.test_tensor_size import display_tensor



class VideoArticulatoryEncoder(nn.Module):
    def __init__(self,input_dim = 1024, hidden_dim=256):
        super().__init__()
        # Adaptive pooling to handle variable input length
        self.g_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
            # nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )

    def forward(self, video_features):
        video_features_prime = self.g_transform(video_features)
        return video_features_prime
        # Handle 4D input by collapsing first two dimensions
# va = VideoArticulatoryEncoder()
#
# device = CONFIG.get("device")
# # model = va.to(device)
# features_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_1_video_embeddings/batch_features_lips.pt"
# video_features= torch.load(features_path)
#
# print(video_features.size())
# video_features_prime = va(video_features)
# print(video_features_prime.size())
# display_tensor(video_features_prime)
# display_tensor(video_features)