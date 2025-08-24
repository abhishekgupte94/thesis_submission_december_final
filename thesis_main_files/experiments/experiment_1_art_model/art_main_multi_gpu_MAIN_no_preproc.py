# === MODIFICATION NOTE (2025-08-23) ===
# Preprocessing CLI functionality removed.
# - Commented out argparse arguments related to preprocessing.
# - Replaced any 'args.<...preprocess...>' checks with False so those paths never run.
# - Commented out direct calls to functions whose names include 'preprocess'.
# - Original lines remain in-place, commented and annotated with [REMOVED PREPROCESSING].
# =======================================

# Ensure the root of the project is in sys.path
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from codecarbon import EmissionsTracker

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
from thesis_main_files.models.data_loaders.data_loader_ART import (
    VideoAudioDataset,
    VideoAudioFeatureProcessor,
    convert_paths_for_training,
    convert_paths_for_evaluation,
    preprocess_videos_before_training,  # not used; kept for reference
    get_project_root,
)
from thesis_main_files.models.art_avdf.training_pipeline.training_ART_Multi_GPU import TrainingPipeline
from thesis_main_files.main_files.evaluation.art.evaluator import (
    evaluate_model_standalone,  # Standalone convenience function (evaluator logic unchanged)
)
from thesis_main_files.main_files.evaluation.art.evaluator import evaluate_model_standalone
from thesis_main_files.models.art_avdf.evaluation_pipeline.evaluating_final_model import EvaluationPipeline


def setup_gpu_allocation(strategy="dedicated_video"):
    """
    Simple GPU allocation setup
    Returns: (video_gpu, audio_gpu, training_gpus_str)
    """
    if strategy == "dedicated_video":
        # GPU 0: Video, GPU 1: Audio, GPUs 2-7: Training
        return 0, 1, "2,3,4,5,6,7"
    elif strategy == "shared_video":
        # GPU 0: Video+Audio, GPUs 1-7: Training
        return 0, 0, "1,2,3,4,5,6,7"
    else:
        # Default: everything on available GPUs
        return 0, 0, "0,1,2,3,4,5,6,7"
# -------------------------------------------------------------------
# UPDATED: Resolve dataset paths based on mode flags (train/evaluate)
# - For training: returns (csv_path, video_dir)
# - For evaluation: returns (fake_csv_path, fake_video_dir, real_csv_path, real_video_dir)
# -------------------------------------------------------------------
def _resolve_dataset_paths(args):
    """
    Resolve dataset paths depending on whether we're training or evaluating.

    For training:
        Uses convert_paths_for_training(args.csv_file).
        Returns (csv_path, video_dir)

    For evaluation:
        Uses convert_paths_for_evaluation(args.csv_file).
        Returns (fake_csv_path, fake_video_dir, real_csv_path, real_video_dir)
    """
    if args.train:
        csv_path, video_dir = convert_paths_for_training(args.csv_file)
        return csv_path, video_dir

    if args.evaluate:
        fake_csv_path, fake_video_dir, real_csv_path, real_video_dir = convert_paths_for_evaluation(args.csv_file)
        return fake_csv_path, fake_video_dir, real_csv_path, real_video_dir

    raise ValueError("Either args.train or args.evaluate must be set to True.")


def main():
    # 1) Argument parser
    parser = argparse.ArgumentParser(description="Train or Evaluate ART model (multi-GPU ready)")

# [REMOVED PREPROCESSING ARGS]     parser.add_argument('--preprocess', action='store_true', help='Only preprocess videos and exit')
# [REMOVED PREPROCESSING ARGS]     parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU or per video batch in preprocessing')
# [REMOVED PREPROCESSING ARGS]     parser.add_argument('--csv_file', type=str, default="training_data_two.csv", help='CSV filename for training or preprocessing')

    # Runtime essentials
    parser.add_argument('--csv_file', type=str, default=None, help='CSV filename for dataset split')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU')

    # Mode switches
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation and exit')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional checkpoint to load for evaluation')
    # NEW: Add GPU allocation arguments
    parser.add_argument('--gpu_strategy', type=str, default='dedicated_video',
                        choices=['dedicated_video', 'shared_video', 'default'],
                        help='GPU allocation strategy')
    parser.add_argument('--verbose_gpu', action='store_true',
                        help='Print GPU allocation info')
    args = parser.parse_args()

    # Default behavior: if neither flag is set, run training (preserves previous behavior)
    if not args.train and not args.evaluate:
        args.train = True

    # CSV defaults depending on mode (requested behavior)
    if args.train and args.csv_file is None:
        args.csv_file = "sample_real_70_percent_half1.csv"
    elif args.evaluate and args.csv_file is None:
        args.csv_file = "training_data_two.csv"

    # 2) Project paths
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("save_final_model", exist_ok=True)
    os.makedirs("carbon_logs_preprocessing", exist_ok=True)
    # NEW: add dedicated logs for training/eval
    os.makedirs("carbon_logs_training", exist_ok=True)
    os.makedirs("carbon_logs_eval", exist_ok=True)

    batch_size = args.batch_size
    # NEW: Setup GPU allocation
    video_gpu, audio_gpu, training_gpus = setup_gpu_allocation(args.gpu_strategy)

    if args.verbose_gpu or args.train:  # Always show for training
        print(f"🔧 GPU Strategy: {args.gpu_strategy}")
        print(f"📺 Video extraction: GPU {video_gpu}")
        print(f"🔊 Audio extraction: GPU {audio_gpu}")
        print(f"🏃 Training GPUs: {training_gpus}")

    # -------------------------------------------------------------------
    # STANDALONE EVALUATION MODE (no training)
    # -------------------------------------------------------------------
    if args.evaluate and not args.train:
        fake_csv_path, fake_video_dir, real_csv_path, real_video_dir = _resolve_dataset_paths(args)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # NEW -  Use dedicated video GPU for evaluation
        device = torch.device(f"cuda:{video_gpu}")
        # dataset + feature_processor: SAME pattern as training
        dataset = VideoAudioDataset(csv_path=fake_csv_path, video_dir=fake_video_dir)
        # feature_processor = VideoAudioFeatureProcessor(batch_size=args.batch_size)
        # NEW Create feature processor with dedicated GPUs
        feature_processor = VideoAudioFeatureProcessor(
            batch_size=args.batch_size,
            video_gpu_id=video_gpu,
            audio_gpu_id=audio_gpu,
            verbose=args.verbose_gpu
        )
        evaluator = EvaluationPipeline(
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            feature_processor=feature_processor,
            K=50,
            common_dim=512,
            checkpoint_path=args.checkpoint,  # pass your saved model here
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            print_every=1,
            use_tsne=True,
            use_retrieval_plot=True,
            t_sne_save_path="eval_tsne.png",
            retrieval_save_path="eval_retrieval.png",
        )

        # === CodeCarbon: start tracking for EVALUATION ===
        tracker = EmissionsTracker(
            project_name="ssl_project_eval",
            output_dir="carbon_logs_eval",
            save_to_file=True,
        )
        tracker.start()
        try:
            results = evaluator.run(k_list=[1, 5])
        finally:
            emissions = tracker.stop()  # kg CO2eq
            print(f"🌱 CodeCarbon (eval) emissions: {emissions:.6f} kg CO2eq")

        print("✅ Evaluation finished.")
        print("Recall@1:", f"{results['recall_at_k']['R@1']:.4f}")
        if 'R@5' in results['recall_at_k']:
            print("Recall@5:", f"{results['recall_at_k']['R@5']:.4f}")
        return

    # -------------------------------------------------------------------
    # TRAINING MODE (gated behind --train)
    # -------------------------------------------------------------------
    if args.train:
        # Set CUDA_VISIBLE_DEVICES for training processes
        os.environ["CUDA_VISIBLE_DEVICES"] = training_gpus
        print(f"🎯 Training will use GPUs: {training_gpus}")
        # Reflect the new _resolve_dataset_paths(args) shape:
        # For training we get 2-tuple: (csv_path, video_dir)
        csv_path, video_dir = _resolve_dataset_paths(args)

        # ---------- TRAINING FLOW (DDP) ----------
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

        # Load dataset with provided csv
        dataset = VideoAudioDataset(csv_path=csv_path, video_dir=video_dir)

        # Feature processor
        # feature_processor = VideoAudioFeatureProcessor(batch_size=batch_size)
        # NEW Create feature processor with dedicated GPUs
        # NOTE: video_gpu and audio_gpu are PHYSICAL GPU IDs (outside CUDA_VISIBLE_DEVICES)
        feature_processor = VideoAudioFeatureProcessor(
            batch_size=batch_size,
            video_gpu_id=video_gpu,  # Physical GPU 0
            audio_gpu_id=audio_gpu,  # Physical GPU 1
            verbose=(args.verbose_gpu and local_rank == 0)
        )
        # Initialize trainer
        trainer = TrainingPipeline(
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=1e-4,
            num_epochs=150,
            device=torch.device(f"cuda:{local_rank}"),
            feature_processor=feature_processor,
            output_txt_path=None,
            local_rank=local_rank
        )

        # === CodeCarbon: start tracking for TRAINING (per-rank logs) ===
        train_log_dir = os.path.join("carbon_logs_training", f"rank_{local_rank}")
        os.makedirs(train_log_dir, exist_ok=True)
        tracker = EmissionsTracker(
            project_name="ssl_project_train",
            output_dir=train_log_dir,
            save_to_file=True,
            # If you prefer to log only on rank 0, move tracker construction
            # under `if dist.get_rank() == 0:` and set train_log_dir = "carbon_logs_training".
        )

        tracker.start()
        try:
            # Start training
            trainer.train("checkpoint/")
            trainer.save_final_state("save_final_model/final_model.pt")
        finally:
            emissions = tracker.stop()  # kg CO2eq
            # Print from each rank (useful if you keep per-rank tracking)
            print(f"🌱 CodeCarbon (train, rank {local_rank}) emissions: {emissions:.6f} kg CO2eq")

        # Cleanup
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
