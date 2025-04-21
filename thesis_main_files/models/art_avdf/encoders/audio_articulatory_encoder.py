#
# import torch
# import torch.nn as nn
# class AudioArticulatoryEncoder(nn.Module):
#     def __init__(self, input_dim=1473, hidden_dim = 128):
#         super().__init__()
#         # self.input_dim = 128
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         # Project 2093 features to hidden_dim
#         self.input_projection = nn.Linear(input_dim, hidden_dim)  # [1, 2093] -> [1, 128]
#         # Single feed-forward attention layer
#         self.feed_forward_attention = FeedForwardAttention(hidden_dim, num_heads=8)
#
#         # # Style encoder for single AdaIN block
#         # self.style_encoder = nn.Sequential(
#         #     nn.Linear(hidden_dim, 64),
#         #     nn.ReLU(),
#         #     nn.LayerNorm(64)
#         # )
#
#         # Single AdaIN block
#         # self.adain_block = AdaINBlock(hidden_dim, style_dim=64)
#
#         # Bi-LSTM for temporal modeling
#         # self.bilstm = nn.LSTM(
#         #     input_size=hidden_dim,
#         #     hidden_size=hidden_dim // 2,
#         #     bidirectional=True,
#         #     batch_first=True,
#         #     num_layers=2
#         # )
#
#         self.dropout = nn.Dropout(0.1)
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#     def forward(self, features):
#         # features = features.transpose(0, 1)  # [2093, 1]
#         features = self.input_projection(features)  # [2093, 128]        # Process through feed-forward attention
#         # for attention in self.feed_forward_attention:
#         features = self.feed_forward_attention(features)
#         features = self.dropout(features)
#
#         # Generate style features
#         # style = self.style_encoder(features)
#         #
#
#         # features = self.adain_block(features, style)
#         features = self.layer_norm(features)
#
#         # Temporal modeling through Bi-LSTM
#         # features, _ = self.bilstm(features.unsqueeze(0))  # Add batch dimension
#         # features = self.dropout(features)
#         return features
#
#
# class FeedForwardAttention(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.attention1 = nn.MultiheadAttention(dim, num_heads)
#         self.norm1 = nn.LayerNorm(dim)
#         self.ffn1 = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(dim * 4, dim)
#         )
#         self.attention2 = nn.MultiheadAttention(dim, num_heads)
#         self.norm2 = nn.LayerNorm(dim)
#         self.ffn2 = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(dim * 4, dim)
#         )
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         residual = x
#         x = self.norm1(x)
#         attn_output, _ = self.attention1(x, x, x)
#         x = residual + self.dropout(attn_output)
#         x = x + self.dropout(self.ffn1(x))
#         residual = x
#         x = self.norm2(x)
#         attn_output, _ = self.attention2(x, x, x)
#         x = residual + self.dropout(attn_output)
#         x = x + self.dropout(self.ffn2(x))
#         return x
#
#
# class AdaINBlock(nn.Module):
#     def __init__(self, dim, style_dim=64):
#         super().__init__()
#         self.style_layer = nn.Linear(style_dim, dim * 2)
#         self.norm = nn.InstanceNorm1d(dim, affine=False)
#         self.fc = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(dim * 2, dim * 2)
#         )
#
#     def forward(self, x, style):
#         style_params = self.style_layer(style)
#         s_gamma, s_beta = style_params.chunk(2, dim=-1)
#         normalized = self.norm(x.transpose(1, 2)).transpose(1, 2)
#         params = self.fc(x)
#         gamma, beta = params.chunk(2, dim=-1)
#         out = normalized * (1 + gamma * s_gamma) + (beta + s_beta)
#         return out
#
# # from torchsummary import summary
# # # or
# # # from torchinfo import summary
# # #
# # # # Create model instance
# # # encoder = AudioArticulatoryEncoder(input_dim=2093)
# # #
# # # # Print model summary
# # # # For torchsummary:
# # # # summary(encoder, input_size=(1, 2093))
# # #
# # # # Or for torchinfo (provides more detailed output):
# # # summary(encoder, input_size=(1, 2093), device='cpu')

import torch
import torch.nn as nn
features_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/audio_wav/concatenated"


class AudioArticulatoryEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Adaptive pooling to handle variable input length
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, hidden_dim))  # Changed to 2D pooling

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.adain = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, features):
        # Handle 4D input by collapsing first two dimensions
        B = features.size(0)
        features = features.view(B, -1, features.size(-1))  # [B, C*H, W]

        # Apply 2D adaptive pooling
        x = self.adaptive_pool(features.unsqueeze(1))  # [B, 1, 1, hidden_dim]
        x = x.squeeze(1).squeeze(1)  # [B, hidden_dim]

        # Apply normalization and processing
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.adain(x)

        return x

#
# audio_paths = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_1_audio_embeddings/test_1_audio_embeddings.pt"
# audio_batch = torch.load(audio_paths)
# audio_encoder = AudioArticulatoryEncoder()
# x = audio_encoder(audio_batch)
# print(x.size())

# torch.save(x,"/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_1_audio_embeddings/save_embeddings.pt")

