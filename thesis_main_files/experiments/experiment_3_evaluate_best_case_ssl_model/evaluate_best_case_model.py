import os
import sys
import argparse
import pandas as pd
import torch
from pathlib import Path

from thesis_main_files.models.data_loaders.data_loader_ART import (
    VideoAudioFeatureProcessor,
    get_project_root,
)
from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass
# Preprocessing for evaluation
from thesis_main_files.utils.files_imp import preprocess_videos_for_evaluation
# Your model
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule

# Util: Load inference CSV
def load_inference_data(csv_path):
    df = pd.read_csv(csv_path)
    video_paths = df['filename'].tolist()
    labels = df['label'].tolist()
    return video_paths, labels

# Util: Load model
def load_model(model_path, model_class):
    model = model_class()
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# 🛠 PATCH Evaluator's evaluation function properly
def patched_evaluate_after_training(self, model, video_paths, labels, preprocess_output_dir,
                 batch_size=128, t_sne_save_path=None, retrieval_save_path=None):
    """
    Batched evaluation, respecting training loop logic.
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

            # Important: Create manifest for this batch
            from thesis_main_files.utils.files_imp import create_manifest_from_selected_files
            create_manifest_from_selected_files(batch_video_paths, self.output_txt_path)

            # Feature extraction
            processed_audio_features, processed_video_features = self.feature_processor.create_datasubset(
                csv_path=None,
                use_preprocessed=False,
                video_paths=batch_video_paths
            )

            if processed_audio_features is None or processed_video_features is None:
                print(f"⚠️ Skipping batch {i//batch_size} due to feature extraction failure.")
                continue

            processed_audio_features = processed_audio_features.to(self.device)
            processed_video_features = processed_video_features.to(self.device)

            f_art, f_lip = model(
                audio_features=processed_audio_features,
                video_features=processed_video_features
            )

            all_f_art.append(f_art)
            all_f_lip.append(f_lip)

    if len(all_f_art) == 0 or len(all_f_lip) == 0:
        print("❌ No valid features extracted for evaluation.")
        return

    # Stack all
    f_art_all = torch.cat(all_f_art, dim=0)
    f_lip_all = torch.cat(all_f_lip, dim=0)

    # Similarity
    similarity_matrix = self.compute_similarity(f_art_all, f_lip_all)

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

# 🚀 Main Runner
def main():
    parser = argparse.ArgumentParser(description="Evaluate SSL model on inference dataset or preprocess only")
    parser.add_argument('--preprocess', action='store_true', help='Only preprocess videos and exit')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to inference CSV file')
    parser.add_argument('--model_checkpoint', type=str, required=False, help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--preprocess_output_dir', type=str, required=True, help='Directory to save preprocessed data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation and preprocessing')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run evaluation')
    parser.add_argument('--t_sne_save_path', type=str, default=None, help='Optional path to save t-SNE plot')
    parser.add_argument('--retrieval_save_path', type=str, default=None, help='Optional path to save retrieval heatmap')
    args = parser.parse_args()

    # Load data
    video_paths, labels = load_inference_data(args.csv_path)

    if args.preprocess:
        print("🔁 Preprocessing videos only...")
        preprocess_videos_for_evaluation(
            video_paths=video_paths,
            output_dir=args.preprocess_output_dir,
            batch_size=args.batch_size
        )
        print("✅ Preprocessing completed.")
        return

    # If evaluation requested
    if args.model_checkpoint is None:
        raise ValueError("Model checkpoint must be provided unless --preprocess is specified.")

    model = load_model(args.model_checkpoint, ARTModule)  # or your specific model

    # Feature Processor
    feature_processor = VideoAudioFeatureProcessor(
        video_preprocess_dir=args.preprocess_output_dir,
        batch_size=args.batch_size
    )

    # Evaluator
    evaluator = EvaluatorClass(
        rank=0,
        device=args.device,
        feature_processor=feature_processor,
        output_txt_path=args.preprocess_output_dir  # Use output dir for manifest temp files
    )

    # Monkey patch evaluator with the fixed batching logic
    evaluator.evaluate_after_training = patched_evaluate_after_training.__get__(evaluator)

    # Evaluate!
    evaluator.evaluate_after_training(
        model=model,
        video_paths=video_paths,
        labels=labels,
        preprocess_output_dir=args.preprocess_output_dir,
        batch_size=args.batch_size,
        t_sne_save_path=args.t_sne_save_path,
        retrieval_save_path=args.retrieval_save_path
    )

    print("✅ Evaluation completed.")

if __name__ == "__main__":
    main()
