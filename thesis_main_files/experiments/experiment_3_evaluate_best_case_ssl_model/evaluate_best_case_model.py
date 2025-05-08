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
from thesis_main_files.models.data_loaders.data_loader_ART import preprocess_videos_before_evaluation,convert_paths_for_svm_train_preprocess
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule

def load_inference_data(csv_path):
    df = pd.read_csv(csv_path)
    video_paths = df['filename'].tolist()
    labels = df['label'].tolist()
    return video_paths, labels

def load_model(model_path, model_class):
    model = model_class()
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

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

    save_dir = "Evaluator"
    t_sne_save_path = os.path.join(save_dir, "t_sne_save_path")
    retrieval_save_path = os.path.join(save_dir, "retrieval_save_path")

    # Load paths and labels
    video_paths, labels = load_inference_data(args.csv_path)
    csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path = convert_paths_for_svm_train_preprocess()
    batch_size = args.batch_size
    if args.preprocess:
        print("🔁 Preprocessing videos only...")

        preprocess_videos_before_evaluation(
            csv_path=csv_path,
            csv_column="video_file",
            output_dir=video_preprocess_dir,
            batch_size=batch_size
        )
        print("✅ Preprocessing completed.")
        return

    if args.model_checkpoint is None:
        raise ValueError("Model checkpoint must be provided unless --preprocess is specified.")

    model = load_model(args.model_checkpoint, ARTModule)

    feature_processor = VideoAudioFeatureProcessor(
        video_preprocess_dir=video_preprocess_dir,
        batch_size=batch_size
    )

    evaluator = EvaluatorClass(
        rank=0,
        device=args.device,
        feature_processor=feature_processor,
        output_txt_path=real_output_txt_path  # Can be any path for manifest temp files
    )

    evaluator.evaluate_after_training(
        model=model,
        video_paths=video_paths,
        labels=labels,
        # preprocess_output_dir=args.preprocess_output_dir,
        batch_size=args.batch_size,
        t_sne_save_path=t_sne_save_path,
        retrieval_save_path=retrieval_save_path
    )

    print("✅ Evaluation completed.")

if __name__ == "__main__":
    main()
