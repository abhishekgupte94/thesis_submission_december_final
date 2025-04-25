import torch
import torch.nn.functional as F

from thesis_main_files.main_files.evaluation.art.retrieval_matching_for_embeddings import RetrievalEvaluator
from thesis_main_files.main_files.evaluation.art.t_sne import TSNEVisualizer

class EvaluatorClass:
    def __init__(self, rank=0, device=None):
        self.visualizer = TSNEVisualizer()
        self.retrieval = RetrievalEvaluator()
        self.rank = rank
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_similarity(self, f_art, f_lip):
        f_art_norm = F.normalize(f_art, dim=1)
        f_lip_norm = F.normalize(f_lip, dim=1)
        similarity_matrix = torch.matmul(f_lip_norm, f_art_norm.T)
        return similarity_matrix

    def evaluate(self, model, audio_inputs, video_inputs, compute_recall=True,
                 t_sne_save_path=None, retrieval_save_path=None, similarity_matrix=None):

        if similarity_matrix is None:
            raise ValueError("Similarity matrix must be provided during evaluation.")

        if self.rank == 0:  # 🔥 Only rank 0 does evaluation outputs
            with torch.no_grad():
                recall_at_1 = self.retrieval.compute_recall_at_k(similarity_matrix, k=1)
                print(f"Recall@1: {recall_at_1:.4f}")

                self.visualizer.visualize(
                    audio_inputs.detach().cpu().numpy(),
                    video_inputs.detach().cpu().numpy(),
                    save_path=t_sne_save_path
                )

                self.retrieval.plot_similarity_matrix(
                    similarity_matrix.detach().cpu(),
                    save_path=retrieval_save_path
                )

    def evaluate_after_training(self, model, audio_inputs, video_inputs,
                 t_sne_save_path="tsne_path.png", retrieval_save_path="retrieval_path.png"):

        model = model.to(self.device)
        model.eval()

        audio_inputs = audio_inputs.to(self.device)
        video_inputs = video_inputs.to(self.device)

        with torch.no_grad():
            f_dash_art, f_dash_lip = model(audio_inputs, video_inputs)
            similarity_matrix = self.compute_similarity(f_dash_art, f_dash_lip)

            if self.rank == 0:  # 🔥 Only rank 0
                recall_at_1 = self.retrieval.compute_recall_at_k(similarity_matrix, k=1)
                print(f"Recall@1: {recall_at_1:.4f}")

                self.visualizer.visualize(
                    f_dash_art.detach().cpu().numpy(),
                    f_dash_lip.detach().cpu().numpy(),
                    save_path=t_sne_save_path
                )

                self.retrieval.plot_similarity_matrix(
                    similarity_matrix.detach().cpu(),
                    save_path=retrieval_save_path
                )
