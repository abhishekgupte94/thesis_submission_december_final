import torch
import torch.nn.functional as F
import os
from thesis_main_files.main_files.evaluation.art.retrieval_matching_for_embeddings import RetrievalEvaluator
from thesis_main_files.main_files.evaluation.art.t_sne import TSNEVisualizer
from thesis_main_files.utils.files_imp import create_manifest_from_selected_files, preprocess_videos_for_evaluation
import numpy as np
from thesis_main_files.utils.preprocess_before_training import preprocess_videos_before_training, preprocess_videos_before_evaluation
class EvaluatorClass:
    def __init__(self, rank=0, device=None, feature_processor = None, output_txt_path = None):
        self.visualizer = TSNEVisualizer()
        self.retrieval = RetrievalEvaluator()
        self.rank = rank
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_processor = feature_processor
        self.output_txt_path = output_txt_path
    def compute_similarity(self, f_art, f_lip):
        f_art_norm = F.normalize(f_art, dim=1)
        f_lip_norm = F.normalize(f_lip, dim=1)
        similarity_matrix = torch.matmul(f_lip_norm, f_art_norm.T)
        return similarity_matrix

    def evaluate_during_training(self, model, similarity_matrix,
                                 audio_inputs, video_inputs,
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
            print("❌ Similarity matrix is None. Skipping evaluation.")
            return

        if self.rank != 0:
            return

        with torch.no_grad():
            recall_at_1 = self.retrieval.compute_recall_at_k(similarity_matrix, k=1)
            print(f"[Eval] Recall@1: {recall_at_1:.4f}")

            # Visualize t-SNE using post-SSL audio/video embeddings
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

    def evaluate_after_training(self, model, video_paths, labels, preprocess_output_dir,
                                batch_size=128, t_sne_save_path=None, retrieval_save_path=None):
        """
        Batched evaluation, following training logic: create_manifest, feature extraction, then evaluation.
        """
        assert self.feature_processor is not None, "feature_processor must be set in EvaluatorClass"
        assert len(video_paths) == len(labels), "Mismatch between video_paths and labels"

        if self.rank == 0:
            print("🔁 Preprocessing videos before evaluation...")
            preprocess_videos_for_evaluation(video_paths, preprocess_output_dir, batch_size=batch_size)

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
                create_manifest_from_selected_files(batch_video_paths, self.output_txt_path)

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

                all_f_art.append(f_art)
                all_f_lip.append(f_lip)

        if len(all_f_art) == 0 or len(all_f_lip) == 0:
            print("❌ No valid features extracted for evaluation.")
            return

        # 4. Stack all batches
        f_art_all = torch.cat(all_f_art, dim=0)
        f_lip_all = torch.cat(all_f_lip, dim=0)

        # 5. Compute similarity
        similarity_matrix = self.compute_similarity(f_art_all, f_lip_all)

        # 6. Only rank 0 saves plots
        if self.rank == 0:
            recall_at_1 = self.retrieval.compute_recall_at_k(similarity_matrix, k=1)
            print(f"Recall@1: {recall_at_1:.4f}")

            self.visualizer.visualize(
                f_art_all.detach().cpu().numpy(),
                f_lip_all.detach().cpu().numpy(),
                save_path=t_sne_save_path
            )

            self.retrieval.plot_similarity_matrix(
                similarity_matrix.detach().cpu(),
                save_path=retrieval_save_path
            )

    # def encode_and_save_features_for_svm(
    #     self,
    #     model,
    #     video_paths,
    #     labels,
    #     save_path_inference_svm,
    #     preprocess_output_dir,
    #     batch_size=128
    # ):
    #     """
    #     Preprocess → extract features → encode → save audio/video features and metadata.
    #     """
    #     assert save_path_inference_svm is not None, "save_path_inference_svm must be specified"
    #     assert self.feature_processor is not None, "feature_processor must be set in EvaluatorClass"
    #     assert len(video_paths) == len(labels), "Mismatch between video_paths and labels"
    #
    #     if self.rank == 0:
    #         print("🔁 Preprocessing videos before feature extraction...")
    #         preprocess_videos_for_evaluation(video_paths, preprocess_output_dir, batch_size=batch_size)
    #     torch.distributed.barrier()  # Sync across all ranks before continuing
    #
    #     model = model.to(self.device)
    #     model.eval()
    #
    #     audio_output_dir = os.path.join(save_path_inference_svm, "audio")
    #     video_output_dir = os.path.join(save_path_inference_svm, "video")
    #     metadata_path = os.path.join(save_path_inference_svm, "metadata.csv")
    #
    #     os.makedirs(audio_output_dir, exist_ok=True)
    #     os.makedirs(video_output_dir, exist_ok=True)
    #     os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    #
    #     metadata = []
    #     sample_idx = 0
    #
    #     with torch.no_grad():
    #         audio_feats, video_feats = self.feature_processor.create_datasubset(
    #             csv_path=None,
    #             use_preprocessed=False,
    #             video_paths=video_paths
    #         )
    #
    #         if audio_feats is None or video_feats is None:
    #             print("❌ Feature extraction failed. Exiting.")
    #             return
    #
    #         audio_feats = audio_feats.to(self.device)
    #         video_feats = video_feats.to(self.device)
    #
    #         f_art, f_lip = model(audio_features=audio_feats, video_features=video_feats)
    #
    #         for i in range(f_art.size(0)):
    #             audio_feat = f_art[i].cpu().numpy()
    #             video_feat = f_lip[i].cpu().numpy()
    #             label = labels[i]
    #
    #             audio_path = os.path.join(audio_output_dir, f"audio_feat_{sample_idx}.npy")
    #             video_path = os.path.join(video_output_dir, f"video_feat_{sample_idx}.npy")
    #
    #             np.save(audio_path, audio_feat)
    #             np.save(video_path, video_feat)
    #
    #             metadata.append((audio_path, video_path, label))
    #             sample_idx += 1
    #
    #     with open(metadata_path, "w") as f:
    #         for audio_path, video_path, label in metadata:
    #             f.write(f"{audio_path},{video_path},{label}\n")
    #
    #     if self.rank == 0:
    #         print(f"[✓] Saved {sample_idx} encoded features")
    #         print(f"[✓] Metadata saved to: {metadata_path}")
    #

