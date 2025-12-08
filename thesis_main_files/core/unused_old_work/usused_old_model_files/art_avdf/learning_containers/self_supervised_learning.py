import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfSupervisedLearning(nn.Module):
    def __init__(self, hidden_dim=128, initial_temperature=0.1):
        super().__init__()
        # Transform lip features via non-linear transformation g(·)
        self.g_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        self.initial_temperature = initial_temperature

    def adjust_temperature(self, f_art, f_lip):
        """
        Dynamically adjusts the temperature based on feature variance.
        Lower variance indicates noisier features, requiring a higher temperature.
        """
        art_var = torch.var(f_art, dim=1).mean()
        lip_var = torch.var(f_lip, dim=1).mean()
        avg_var = (art_var + lip_var) / 2
        # Scale temperature inversely proportional to the variance
        dynamic_temperature = self.initial_temperature * (1 / (avg_var + 1e-6))
        return dynamic_temperature

    def forward(self, f_art, f_lip):
        # Transform lip features to match articulatory space (obtain f'_l)
        # f_lip_prime = self.g_transform(f_lip)

        # Normalize features for cosine similarity
        f_art_norm = F.normalize(f_art, p=2, dim=1)
        f_lip_prime_norm = F.normalize(f_lip, p=2, dim=1)

        # Adjust temperature dynamically
        temperature = self.adjust_temperature(f_art, f_lip_prime_norm)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(f_lip_prime_norm, f_art_norm.T)
        B = f_art.size(0)
        mask = torch.eye(B, device=f_art.device).bool()

        pos_sim = torch.diagonal(similarity_matrix)
        neg_sim = similarity_matrix[~mask].view(B, -1)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temperature
        labels = torch.zeros(B, dtype=torch.long, device=f_art.device)
        loss = F.cross_entropy(logits, labels)
        # f_art_prime = f_art  # Modify if further refinement is needed

        # Return loss, and the transformed features
        return loss,similarity_matrix


