from thesis_main_files.main_files.evaluation.art.retrieval_matching_for_embeddings import RetrievalEvaluator
from thesis_main_files.main_files.evaluation.art.t_sne import TSNEVisualizer
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
import torch
# Evaluation
import torch.nn.functional as F

class EvaluatorClass:
    def __init__(self, rank=0):
        self.visualizer = TSNEVisualizer()
        self.retrieval = RetrievalEvaluator()
        self.rank = rank

    def compute_similarity(self,f_art, f_lip):
        """
        Computes a cosine similarity matrix between two sets of embeddings.

        Args:
            f_art (torch.Tensor): (N, D) - embeddings from articulatory/audio encoder
            f_lip (torch.Tensor): (N, D) - embeddings from lip/video encoder

        Returns:
            torch.Tensor: (N, N) cosine similarity matrix
        """
        f_art_norm = F.normalize(f_art, dim=1)
        f_lip_norm = F.normalize(f_lip, dim=1)
        similarity_matrix = torch.matmul(f_lip_norm, f_art_norm.T)  # [N, N]

        return similarity_matrix

    def evaluate(self, model, audio_inputs, video_inputs, compute_recall=True, t_sne_save_path = None,
                 retrieval_save_path =None):
        model.eval()
        with torch.no_grad():
            f_dash_art, f_dash_lip = model(audio_inputs, video_inputs)
            similarity_matrix = self.compute_similarity(f_dash_art, f_dash_lip)

            if self.rank == 0:
                recall_at_1 = self.retrieval.compute_recall_at_k(similarity_matrix, k=1)
                print(f"Recall@1: {recall_at_1:.4f}")
                self.visualizer.visualize(f_dash_art.cpu().numpy(), f_dash_lip.cpu().numpy(),
                                          save_path=t_sne_save_path)
                self.retrieval.plot_similarity_matrix(similarity_matrix, save_path=retrieval_save_path)
