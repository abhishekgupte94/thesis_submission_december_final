# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class AVDFLoss(nn.Module):
#     def __init__(self, alpha=0.5, lambda_theta=1.0):
#         super().__init__()
#         self.alpha = alpha
#         self.lambda_theta = lambda_theta
#
#     def forward(self, Fva, y):
#         """
#         Args:
#             Fva: Final audio-visual features after fusion
#             y: Overall binary label (0 for real, 1 for fake)
#         """
#         # Cross-entropy loss from paper:
#         # Lce = -Σ(yk · log(exp(fva^k)/Σexp(fva^c)))
#         ce_loss = F.cross_entropy(Fva, y)
#
#         # Contrastive loss from paper:
#         # Lcon = 1/B^2 Σ[Σ(1-sim(Fva^k,Fva^g)) + Σmax(sim(Fva^k,Fva^g)-α,0)]
#         batch_size = Fva.size(0)
#         Fva_norm = F.normalize(Fva, dim=1)
#         sim_matrix = torch.matmul(Fva_norm, Fva_norm.t())
#
#         # Create mask for same/different labels
#         label_matrix = y.unsqueeze(0) == y.unsqueeze(1)
#
#         con_loss = 0
#         for k in range(batch_size):
#             # Same label pairs
#             same_label = sim_matrix[k][label_matrix[k]]
#             same_loss = torch.sum(1 - same_label)
#
#             # Different label pairs
#             diff_label = sim_matrix[k][~label_matrix[k]]
#             diff_loss = torch.sum(torch.clamp(diff_label - self.alpha, min=0))
#
#             con_loss += (same_loss + diff_loss)
#
#         con_loss = con_loss / (batch_size * batch_size)
#
#         return ce_loss + self.lambda_theta * con_loss
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class AVDFLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, labels):
        # Cross-entropy for classification
        ce_loss = F.cross_entropy(output, labels)

        # Simple contrastive loss
        features = F.normalize(output, dim=1)
        sim_matrix = torch.matmul(features, features.t())

        # Create mask for positive/negative pairs
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Compute contrastive loss
        pos_pairs = sim_matrix[mask].mean()
        neg_pairs = sim_matrix[~mask].mean()
        con_loss = -pos_pairs + neg_pairs

        return ce_loss + self.alpha * con_loss