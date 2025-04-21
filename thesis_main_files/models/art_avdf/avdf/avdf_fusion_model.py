# import torch
# import torch.nn as nn
#
#
# class AudioVisualFusion(nn.Module):
#     def __init__(self, hidden_dim=128):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#
#         # Cross-modal attention module
#         self.mha = nn.MultiheadAttention(hidden_dim, num_heads=8)
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(0.1)
#
#         # Feed-forward network for after attention
#         self.ff = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim * 4, hidden_dim)
#         )
#
#     def forward(self, fa, fv, f_art, f_lip_prime):
#         # Expected input shapes:
#         # fa: [Ta × da] where da=128 by default (audio features from VGGish)
#         # fv: [dv × H × W] (visual features from Swin Transformer)
#         # f_art: [T × hidden_dim] (articulatory features from ART module)
#         # f_lip_prime: [T × D] (transformed lip features)
#
#         # Form visual stream by concatenating visual and transformed lip features
#         Fv = torch.cat([fv, f_lip_prime], dim=-1)
#
#         # Form audio stream by concatenating audio and articulatory features
#         Fa = torch.cat([fa, f_art], dim=-1)
#
#         # Directional attention for visual query
#         v1 = self.mha(Fv, Fa, Fa)[0]
#         v2 = self.layer_norm(v1 + Fv)
#         Fv_out = self.layer_norm(self.ff(self.dropout(v2)) + v2)
#
#         # Directional attention for audio query
#         a1 = self.mha(Fa, Fv, Fv)[0]
#         a2 = self.layer_norm(a1 + Fa)
#         Fa_out = self.layer_norm(self.ff(self.dropout(a2)) + a2)
#
#         # Final audio-visual representation
#         Fva = torch.cat([Fv_out, Fa_out], dim=-1)
#
#         return Fva


# class AudioVisualFusion(nn.Module):
#     def __init__(self, hidden_dim=128):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#         self.fc = nn.Linear(hidden_dim * 2, 2)  # Binary classification
#
#     def forward(self, fa, fv, f_art, f_lip):
#         # Concatenate features
#         Fa = torch.cat([fa, f_art], dim=-1)
#         Fv = torch.cat([fv, f_lip], dim=-1)
#
#         # Cross-modal attention
#         x = self.attention(Fa, Fv, Fv)[0]
#         x = self.layer_norm(x)
#
#         # Final classification
#         output = self.fc(torch.cat([x, Fa], dim=-1))
#         return output
import torch
import torch.nn as nn

class AudioVisualFusion(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # Binary classification

    def forward(self, fa, fv, f_art, f_lip_prime):
# Form visual stream by concatenating visual and transformed lip features
        Fv = torch.cat([fv, f_lip_prime], dim=-1)

        # Form audio stream by concatenating audio and articulatory features
        Fa = torch.cat([fa, f_art], dim=-1)

        # Directional attention for visual query
        v1 = self.mha(Fv, Fa, Fa)[0]
        v2 = self.layer_norm(v1 + Fv)
        Fv_out = self.layer_norm(self.ff(self.dropout(v2)) + v2)

        # Directional attention for audio query
        a1 = self.mha(Fa, Fv, Fv)[0]
        a2 = self.layer_norm(a1 + Fa)
        Fa_out = self.layer_norm(self.ff(self.dropout(a2)) + a2)

        # Final audio-visual representation
        Fva = torch.cat([Fv_out, Fa_out], dim=-1)

        return Fva