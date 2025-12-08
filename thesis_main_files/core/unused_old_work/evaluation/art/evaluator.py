
import torch
from thesis_main_files.main_files.unused_old_work.evaluation.art.retrieval_matching_for_embeddings import RetrievalEvaluator
from thesis_main_files.main_files.evaluation.art.t_sne import TSNEVisualizer


# from pathlib import Path
class EvaluatorClass:
    def __init__(self, rank=0, device=None, feature_processor = None, output_txt_path = None):
        self.visualizer = TSNEVisualizer()
        self.retrieval = RetrievalEvaluator()
        self.rank = rank
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_processor = feature_processor
        self.output_txt_path = output_txt_path

    def compute_similarity(self, f_art, f_lip):
        """
        Compute cosine similarity matrix between audio and video features.

        Args:
            f_art (Tensor): Audio feature embeddings (N x D)
            f_lip (Tensor): Video feature embeddings (N x D)

        Returns:
            Tensor: Cosine similarity matrix (N x N)
        """
        f_art = torch.nn.functional.normalize(f_art, p=2, dim=1)
        f_lip = torch.nn.functional.normalize(f_lip, p=2, dim=1)
        similarity_matrix = torch.matmul(f_art, f_lip.t())
        return similarity_matrix

    def evaluate_during_training(self, model, similarity_matrix,
                                 audio_inputs=None, video_inputs=None,
                                 t_sne_save_path=None, retrieval_save_path=None):
        """
        Evaluate the model during training using similarity matrix, t-SNE, and retrieval heatmap.

        Args:
            model (nn.Module): The model being trained.
            similarity_matrix (Tensor): Cosine similarity matrix between audio and video features.
            audio_inputs (Tensor): f_art embeddings from the model.
            video_inputs (Tensor): f_lip embeddings from the model.
            t_sne_save_path (str, optional): File path to save t-SNE plot.
            retrieval_save_path (str, optional): File path to save retrieval heatmap.
        """
        if similarity_matrix is None:
            raise ValueError("similarity_matrix is required for evaluate_during_training")

        if self.rank == 0:
            recall_at_1 = self.retrieval.compute_recall_at_k(similarity_matrix, k=1)
            print(f"[Eval/Train] Recall@1: {recall_at_1:.4f}")

            if audio_inputs is not None and video_inputs is not None:
                self.visualizer.visualize(
                    audio_inputs.detach().cpu().numpy(),
                    video_inputs.detach().cpu().numpy(),
                    save_path=t_sne_save_path
                )

            self.retrieval.plot_similarity_matrix(
                similarity_matrix.detach().cpu(),
                save_path=retrieval_save_path
            )

    def evaluate_after_training(self, model, video_paths, labels,
                                batch_size=128, t_sne_save_path=None, retrieval_save_path=None):
        """
        Batched evaluation, following training logic: create_manifest, feature extraction, then evaluation.
        """
        assert self.feature_processor is not None, "feature_processor must be set in EvaluatorClass"
        assert len(video_paths) == len(labels), "Mismatch between video_paths and labels"

        # if self.rank == 0:
        #     print("🔁 Preprocessing video before evaluation...")
        #     preprocess_videos_for_evaluation(video_paths, preprocess_output_dir, batch_size=batch_size)

        model = model.to(self.device)
        model.eval()

        all_f_art = []
        all_f_lip = []

        with torch.no_grad():
            # Break into mini-batches manually
            for i in range(0, len(video_paths), batch_size):
                batch_video_paths = video_paths[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                # 1. Very Important: Create manifest for this batch
                # create_manifest_from_selected_files(batch_video_paths, self.output_txt_path)

                # 2. Feature extraction
                processed_audio_features, processed_video_features = self.feature_processor.create_datasubset(
                    csv_path=None,
                    use_preprocessed=False,
                    video_paths=batch_video_paths
                )

                if processed_audio_features is None or processed_video_features is None:
                    print(f"⚠️ Skipping batch {i // batch_size} due to feature extraction failure.")
                    continue

                processed_audio_features = processed_audio_features.to(self.device)
                processed_video_features = processed_video_features.to(self.device)

                # 3. Pass through model
                f_art, f_lip = model(
                    audio_features=processed_audio_features,
                    video_features=processed_video_features
                )

                if isinstance(f_art, torch.Tensor):
                    all_f_art.append(f_art.detach().to(self.device))
                if isinstance(f_lip, torch.Tensor):
                    all_f_lip.append(f_lip.detach().to(self.device))

        if not all_f_art or not all_f_lip:
            raise RuntimeError("No features were produced by feature_processor; check inputs and processor.")

        #         # 1. Very Important: Create manifest for this batch
        #         create_manifest_from_selected_files(
        #             video_paths=batch_video_paths,
        #             labels=batch_labels
        #         )
        #
        #         # 2. Use feature_processor to obtain features
        #         f_art_b, f_lip_b = self.feature_processor.encode_manifest_for_eval(
        #             batch_video_paths,
        #             device=self.device
        #         )
        #
        #         if isinstance(f_art_b, torch.Tensor):
        #             all_f_art.append(f_art_b.detach().to(self.device))
        #         if isinstance(f_lip_b, torch.Tensor):
        #             all_f_lip.append(f_lip_b.detach().to(self.device))
        #
        # if not all_f_art or not all_f_lip:
        #     raise RuntimeError("No features were produced by feature_processor; check inputs and processor.")

        f_art = torch.cat(all_f_art, dim=0)
        f_lip = torch.cat(all_f_lip, dim=0)

        similarity_matrix = self.compute_similarity(f_art, f_lip)

        if self.rank == 0:
            recall_at_1 = self.retrieval.compute_recall_at_k(similarity_matrix, k=1)
            print(f"[Eval] Recall@1: {recall_at_1:.4f}")

            if f_art is not None and f_lip is not None:
                self.visualizer.visualize(
                    f_art.detach().cpu().numpy(),
                    f_lip.detach().cpu().numpy(),
                    save_path=t_sne_save_path
                )

            self.retrieval.plot_similarity_matrix(
                similarity_matrix.detach().cpu(),
                save_path=retrieval_save_path
            )

        return {
            "recall_at_1": float(recall_at_1),
            "similarity_matrix": similarity_matrix.detach().cpu(),
            "f_art": f_art.detach().cpu(),
            "f_lip": f_lip.detach().cpu(),
        }


# ===========================
# ADDED: Standalone wrapper
# (No changes to existing logic. This simply instantiates EvaluatorClass
#  and calls its existing evaluate_after_training method.)
# ===========================
def evaluate_model_standalone(
    model,
    video_paths,
    labels,
    feature_processor,
    *,
    device=None,
    batch_size=128,
    t_sne_save_path=None,
    retrieval_save_path=None,
    rank=0,
    output_txt_path=None
):
    evaluator = EvaluatorClass(
        rank=rank,
        device=device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
        feature_processor=feature_processor,
        output_txt_path=output_txt_path
    )
    return evaluator.evaluate_after_training(
        model=model,
        video_paths=video_paths,
        labels=labels,
        batch_size=batch_size,
        t_sne_save_path=t_sne_save_path,
        retrieval_save_path=retrieval_save_path
    )
