
import torch
import torch.nn as nn

class AudioArticulatoryEncoder(nn.Module):
    def __init__(self, input_dim=64, lstm_hidden_dim=64, output_dim=256, dropout_prob=0.2):
        super(AudioArticulatoryEncoder, self).__init__()

        self.bi_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        lstm_output_dim = 2 * lstm_hidden_dim  # Bi-LSTM has forward + backward

        self.adain = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim * 2),
            nn.ReLU(),
            nn.Linear(lstm_output_dim * 2, lstm_output_dim)
        )

        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.dropout = nn.Dropout(dropout_prob)

        self.proj = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x):
        """
        x: Tensor of shape [B, 64, T] — audio features (melody + prosody)
        Returns:
            out: [B, output_dim] — fixed-size audio embedding
        """
        # Auto-squeeze singleton channel
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)  # [B, C, T]
        x = x.permute(0, 2, 1)  # [B, T, 64]

        lstm_out, _ = self.bi_lstm(x)      # [B, T, 128]
        pooled = lstm_out.mean(dim=1)      # [B, 128]
        styled = self.adain(pooled)        # [B, 128]
        normed = self.layer_norm(styled)   # [B, 128]
        dropped = self.dropout(normed)     # [B, 128]
        out = self.proj(dropped)           # [B, output_dim]

        return out
