# ssl_evaluator_run.py

import os
import sys
import argparse
import pandas as pd
import torch
from pathlib import Path

from thesis_main_files.models.data_loaders.data_loader_ART import (
    VideoAudioFeatureProcessor,
    get_project_root,
    preprocess_videos_before_training
)
from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass

# Example (you should replace this with your real model import)
# from your_model_definition import YourModelClass
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule  # Example

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
    # 1) Argument parser
    parser = argparse.ArgumentParser(description="Evaluate SSL model on inference dataset")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to inference CSV file')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--preprocess_output_dir', type=str, required=True, help='Directory to save preprocessed data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation and preprocessing')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run evaluation')
    parser.add_argument('--t_sne_save_path', type=str, default=None, help='Optional path to save t-SNE plot')
    parser.add_argument('--retrieval_save_path', type=str, default=None, help='Optional path to save retrieval heatmap')
    args = parser.parse_args()

    # 2) Load video paths and labels
    video_paths, labels = load_inference_data(args.csv_path)

    # 3) Load model
    model = load_model(args.model_checkpoint, ARTModule)  # <<< Replace with your model if different

    # 4) Initialize feature processor
    feature_processor = VideoAudioFeatureProcessor(
        video_preprocess_dir=args.preprocess_output_dir,
        batch_size=args.batch_size
    )

    # 5) Initialize Evaluator
    evaluator = EvaluatorClass(
        rank=0,
        device=args.device,
        feature_processor=feature_processor,
        output_txt_path=None  # can be set if needed
    )

    # 6) Start full evaluation
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
