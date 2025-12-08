# # evaluation_pipeline.py
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from typing import Dict, Optional, List
#
# # model
# from thesis_main_files.models.art_avdf.encoders.new_encoder_for_ssl.alignment_pipeline_gpu import GPUOptimizedAlignment
# # your existing evaluator utilities (kept as-is)
# from thesis_main_files.core.evaluation.art.retrieval_matching_for_embeddings import RetrievalEvaluator
# from thesis_main_files.core.evaluation.art.t_sne import TSNEVisualizer
#
#
# class EvaluationPipeline:
#     """
#     Evaluation pipeline that mirrors TrainingPipeline's constructor style.
#     - Takes a dataset (same one you use in training).
#     - Takes the same VideoAudioFeatureProcessor instance.
#     - Iterates in batches exactly like training: each batch yields (video_paths, audio_paths, labels),
#       then feature_processor.create_datasubset(video_paths) produces tensors for the model.
#
#     It accumulates pooled embeddings, computes cosine similarity, and reports Recall@k.
#     """
#
#     def __init__(
#         self,
#         dataset,
#         batch_size: int,
#         device: torch.device,
#         feature_processor,
#         *,
#         K: int = 50,
#         common_dim: int = 512,
#         checkpoint_path: Optional[str] = None,
#         num_workers: int = 8,
#         pin_memory: bool = True,
#         persistent_workers: bool = True,
#         print_every: int = 1,
#         use_tsne: bool = True,
#         use_retrieval_plot: bool = True,
#         t_sne_save_path: Optional[str] = "eval_tsne.png",
#         retrieval_save_path: Optional[str] = "eval_retrieval.png",
#     ):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.device = device
#         self.feature_processor = feature_processor
#         self.print_every = print_every
#
#         # dataloader mirrors training (no sampler needed for single-GPU eval)
#         self.dataloader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             persistent_workers=persistent_workers,
#         )
#
#         # model must match training hyperparams
#         self.model = GPUOptimizedAlignment(rank=0, world_size=1, K=K, common_dim=common_dim).to(device)
#         self.model.eval()
#
#         if checkpoint_path:
#             self._load_checkpoint(checkpoint_path)
#
#         # eval helpers (use your implementations, unchanged)
#         self.retrieval = RetrievalEvaluator()
#         self.visualizer = TSNEVisualizer()
#         self.use_tsne = use_tsne
#         self.use_retrieval_plot = use_retrieval_plot
#         self.t_sne_save_path = t_sne_save_path
#         self.retrieval_save_path = retrieval_save_path
#
#     def _load_checkpoint(self, path: str):
#         print(f"🔄 Loading checkpoint for eval: {path}")
#         state = torch.load(path, map_location=self.device)
#         state_dict = state.get("model_state_dict", state)
#         # self.model.load_state_dict(state_dict, strict=True)
#         self.model.load_state_dict(state_dict, strict=False)
#
#         print("✅ Checkpoint loaded.")
#
#     @torch.no_grad()
#     def run(self, k_list: List[int] = [1]) -> Dict[str, object]:
#         all_audio, all_video = [], []
#         all_labels = []
#
#         total_batches = len(self.dataloader)
#         for b_idx, (video_paths, audio_paths, labels) in enumerate(self.dataloader):
#             # 1) use the SAME processor call as training
#             a_feats, v_feats = self.feature_processor.create_datasubset(
#                 csv_path=None,
#                 use_preprocessed=False,
#                 video_paths=video_paths,  # training passes only video_paths; we mirror that
#             )
#             if a_feats is None or v_feats is None:
#                 print(f"⚠️ Skipping batch {b_idx}: feature extraction failed.")
#                 continue
#
#             a_feats = a_feats.to(self.device, non_blocking=True)  # [b,101,768]
#             v_feats = v_feats.to(self.device, non_blocking=True)  # [b,  8,768]
#
#             # 2) forward through alignment model (dict outputs)
#             out = self.model(audio_features=a_feats, video_features=v_feats)
#             a_seq = out["audio_aligned"]  # [b, K, D]
#             v_seq = out["video_aligned"]  # [b, K, D]
#
#             # 3) pool to [b, D]
#             f_a = a_seq.mean(dim=1).cpu()
#             f_v = v_seq.mean(dim=1).cpu()
#
#             all_audio.append(f_a)
#             all_video.append(f_v)
#             if labels is not None:
#                 # ensure labels is a 1D list/array for bookkeeping
#                 if hasattr(labels, "tolist"):
#                     all_labels.extend(labels.tolist())
#                 else:
#                     all_labels.extend(list(labels))
#
#             if self.print_every and (b_idx % self.print_every == 0):
#                 print(f"[Eval] Batch {b_idx+1}/{total_batches}")
#
#         if not all_audio or not all_video:
#             raise RuntimeError("No eval embeddings produced; check your feature extraction.")
#
#         # 4) concat to [N, D] (CPU)
#         f_audio = torch.cat(all_audio, dim=0)
#         f_video = torch.cat(all_video, dim=0)
#
#         # 5) cosine similarity (normalize first)
#         f_audio = F.normalize(f_audio, p=2, dim=1)
#         f_video = F.normalize(f_video, p=2, dim=1)
#         similarity = f_audio @ f_video.T  # [N, N] on CPU
#
#         # 6) metrics via your RetrievalEvaluator
#         metrics = {}
#         for k in k_list:
#             r_at_k = self.retrieval.compute_recall_at_k(similarity, k=k)
#             metrics[f"R@{k}"] = float(r_at_k)
#
#         print("✅ Evaluation complete:", ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
#
#         # 7) optional visuals
#         if self.use_tsne and self.t_sne_save_path:
#             self.visualizer.visualize(
#                 f_audio.detach().cpu().numpy(),
#                 f_video.detach().cpu().numpy(),
#                 save_path=self.t_sne_save_path,
#             )
#         if self.use_retrieval_plot and self.retrieval_save_path:
#             self.retrieval.plot_similarity_matrix(similarity.detach().cpu(), save_path=self.retrieval_save_path)
#
#         return {
#             "recall_at_k": metrics,
#             "similarity_matrix": similarity,  # CPU tensor
#             "f_audio": f_audio,               # CPU tensor
#             "f_video": f_video,               # CPU tensor
#             "labels": all_labels,
#         }
# evaluation_pipeline.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List

# model
from thesis_main_files.core.unused_old_work.usused_old_model_files.art_avdf.encoders.new_encoder_for_ssl.alignment_pipeline_gpu import GPUOptimizedAlignment
# your existing evaluator utilities (kept as-is)
from thesis_main_files.main_files.unused_old_work.evaluation.art.retrieval_matching_for_embeddings import RetrievalEvaluator
from thesis_main_files.main_files.evaluation.art.t_sne import TSNEVisualizer


class EvaluationPipeline:
    """
    Evaluation pipeline that mirrors TrainingPipeline's constructor style.
    - Takes a dataset (same one you use in training).
    - Takes the same VideoAudioFeatureProcessor instance.
    - Iterates in batches exactly like training: each batch yields (video_paths, audio_paths, labels),
      then feature_processor.create_datasubset(video_paths) produces tensors for the model.

    It accumulates pooled embeddings, computes cosine similarity, and reports Recall@k.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        device: torch.device,
        feature_processor,
        *,
        K: int = 50,
        common_dim: int = 512,
        checkpoint_path: Optional[str] = None,
        num_workers: int = 8,
        # pin_memory: bool = True, # multi-GPU training
        pin_memory: bool = False, # MAC/ARM training
        persistent_workers: bool = True,
        print_every: int = 1,
        use_tsne: bool = True,
        use_retrieval_plot: bool = True,
        t_sne_save_path: Optional[str] = "eval_tsne.png",
        retrieval_save_path: Optional[str] = "eval_retrieval.png",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.feature_processor = feature_processor
        self.print_every = print_every

        # dataloader mirrors training (no sampler needed for single-GPU eval)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        # model must match training hyperparams
        self.model = GPUOptimizedAlignment(rank=0, world_size=1, K=K, common_dim=common_dim).to(device)
        self.model.eval()

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # eval helpers (use your implementations, unchanged)
        self.retrieval = RetrievalEvaluator()
        self.visualizer = TSNEVisualizer()
        self.use_tsne = use_tsne
        self.use_retrieval_plot = use_retrieval_plot
        self.t_sne_save_path = t_sne_save_path
        self.retrieval_save_path = retrieval_save_path

    def _load_checkpoint(self, path: str):
        print(f"ðŸ”„ Loading checkpoint for eval: {path}")
        state = torch.load(path, map_location=self.device)
        state_dict = state.get("model_state_dict", state)
        # self.model.load_state_dict(state_dict, strict=True)
        self.model.load_state_dict(state_dict, strict=False)

        print("âœ… Checkpoint loaded.")

    @torch.no_grad()
    def run(self, k_list: List[int] = [1]) -> Dict[str, object]:
        all_audio, all_video = [], []
        all_labels = []

        total_batches = len(self.dataloader)
        for b_idx, (video_paths, audio_paths, labels) in enumerate(self.dataloader):
            # 1) use the SAME processor call as training
            a_feats, v_feats = self.feature_processor.create_datasubset(
                csv_path=None,
                use_preprocessed=False,
                video_paths=video_paths,  # training passes only video_paths; we mirror that
            )
            if a_feats is None or v_feats is None:
                print(f"âš ï¸ Skipping batch {b_idx}: feature extraction failed.")
                continue

            a_feats = a_feats.to(self.device, non_blocking=True)  # [b,101,768]
            v_feats = v_feats.to(self.device, non_blocking=True)  # [b,  8,768]

            # 2) forward through alignment model (dict outputs)
            out = self.model(audio_features=a_feats, video_features=v_feats)
            a_seq = out["audio_aligned"]  # [b, K, D]
            v_seq = out["video_aligned"]  # [b, K, D]

            # 3) pool to [b, D]
            f_a = a_seq.mean(dim=1).cpu()
            f_v = v_seq.mean(dim=1).cpu()

            all_audio.append(f_a)
            all_video.append(f_v)
            if labels is not None:
                # ensure labels is a 1D list/array for bookkeeping
                if hasattr(labels, "tolist"):
                    all_labels.extend(labels.tolist())
                else:
                    all_labels.extend(list(labels))

            if self.print_every and (b_idx % self.print_every == 0):
                print(f"[Eval] Batch {b_idx+1}/{total_batches}")

        if not all_audio or not all_video:
            raise RuntimeError("No eval embeddings produced; check your feature extraction.")

        # 4) concat to [N, D] (CPU)
        f_audio = torch.cat(all_audio, dim=0)
        f_video = torch.cat(all_video, dim=0)

        # 5) cosine similarity (normalize first)
        f_audio = F.normalize(f_audio, p=2, dim=1)
        f_video = F.normalize(f_video, p=2, dim=1)
        similarity = f_audio @ f_video.T  # [N, N] on CPU

        # 6) metrics via your RetrievalEvaluator
        metrics = {}
        for k in k_list:
            r_at_k = self.retrieval.compute_recall_at_k(similarity, k=k)
            metrics[f"R@{k}"] = float(r_at_k)

        print("âœ… Evaluation complete:", ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

        # 7) optional visuals
        if self.use_tsne and self.t_sne_save_path:
            self.visualizer.visualize(
                f_audio.detach().cpu().numpy(),
                f_video.detach().cpu().numpy(),
                save_path=self.t_sne_save_path,
            )
        if self.use_retrieval_plot and self.retrieval_save_path:
            self.retrieval.plot_similarity_matrix(similarity.detach().cpu(), save_path=self.retrieval_save_path)

        return {
            "recall_at_k": metrics,
            "similarity_matrix": similarity,  # CPU tensor
            "f_audio": f_audio,               # CPU tensor
            "f_video": f_video,               # CPU tensor
            "labels": all_labels,
        }