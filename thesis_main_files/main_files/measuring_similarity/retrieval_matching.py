import torch

class RetrievalEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def compute_recall_at_k(similarity_matrix, k=1):
        """
        Computes Recall@K for a similarity matrix.

        Args:
            similarity_matrix (torch.Tensor): (N, N) similarity between modality A and B
            k (int): Recall threshold (e.g., Recall@1, Recall@5)

        Returns:
            float: Recall@K score (0.0 to 1.0)
        """
        # Sort rows (queries) by descending similarity
        sorted_indices = similarity_matrix.argsort(dim=1, descending=True)  # shape: (N, N)

        # Ground truth matches are on the diagonal: i-th row should match i-th column
        ground_truth = torch.arange(similarity_matrix.size(0)).unsqueeze(1).to(similarity_matrix.device)

        # Get top-k predicted indices
        top_k = sorted_indices[:, :k]  # shape: (N, k)

        # Check if correct match is in top-k
        hits = (top_k == ground_truth).any(dim=1).float()  # shape: (N,)

        # Compute mean recall
        recall = hits.mean().item()
        return recall
