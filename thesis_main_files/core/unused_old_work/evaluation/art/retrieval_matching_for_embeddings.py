import torch
import matplotlib.pyplot as plt
import seaborn as sns

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
        sorted_indices = similarity_matrix.argsort(dim=1, descending=True)
        ground_truth = torch.arange(similarity_matrix.size(0)).unsqueeze(1).to(similarity_matrix.device)
        top_k = sorted_indices[:, :k]
        hits = (top_k == ground_truth).any(dim=1).float()
        recall = hits.mean().item()
        return recall

    @staticmethod
    def plot_similarity_matrix(similarity_matrix, title="Similarity Matrix", save_path=None):
        """
        Plots a heatmap of the similarity matrix and optionally saves it.

        Args:
            similarity_matrix (torch.Tensor): (N, N)
            title (str): Plot title
            save_path (str or None): Path to save the plot (e.g., "similarity_heatmap.png")
        """
        similarity_np = similarity_matrix.cpu().numpy()

        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_np, cmap='viridis', cbar=True)
        plt.title(title)
        plt.xlabel("Video Embeddings")
        plt.ylabel("Audio Embeddings")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Similarity matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()
