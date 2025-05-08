# svm_trainer_run.py

import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from thesis_main_files.models.data_loaders.data_loader_ART import (
    # AudioVideoPathDataset,
    VideoAudioFeatureProcessor,
    convert_paths_for_svm_train_preprocess,
    convert_paths_for_svm_val_preprocess,
    preprocess_videos_before_training,
    get_project_root,
)
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
from thesis_main_files.models.svm.create_data_loader_for_inference import AudioVideoPathDataset
from thesis_main_files.models.svm.create_data_loader_for_svm import FeatureBuilder
from thesis_main_files.models.svm.train_svm_on_deep_features import SVMTrainer
from thesis_main_files.models.svm.evaluate_svm import SVMEvaluator
from thesis_main_files.models.svm.create_data_loader_for_inference import AudioVideoPathDataset
from thesis_main_files.models.svm.create_data_loader_for_svm import FeatureBuilder
from thesis_main_files.models.svm.train_svm_on_deep_features import SVMTrainer
from thesis_main_files.models.svm.evaluate_svm import SVMEvaluator
def main():
    # 1) Argument parser
    parser = argparse.ArgumentParser(description="Train SVM model after feature extraction")
    parser.add_argument('--preprocess', action='store_true', help='Only preprocess videos and exit')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for video preprocessing or feature extraction')
    parser.add_argument('--csv_file', type=str, default="training_data_svm_final.csv", help='CSV file name')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to frozen ART model checkpoint (.pth)')
    parser.add_argument('--save_dir', type=str, default="svm_results", help='Directory to save npz, plots, etc.')
    args = parser.parse_args()

    # 2) Project paths
    os.makedirs(args.save_dir, exist_ok=True)

    # 3) Select correct path function based on CSV
    if args.csv_file == "training_data_svm_final.csv":
        csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path = convert_paths_for_svm_train_preprocess()
    elif args.csv_file == "val_data_for_svm.csv":
        csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path = convert_paths_for_svm_val_preprocess()
    else:
        raise ValueError(f"Unsupported CSV file '{args.csv_file}'. Please use: training_data_svm_final.csv or val_data_for_svm.csv")

    batch_size = args.batch_size

    if args.preprocess:
        # Only preprocess videos
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(project_name="preprocessing_svm", output_dir="carbon_logs_preprocessing")
        tracker.start()

        preprocess_videos_before_training(
            csv_path=csv_path,
            csv_column="video_file",
            output_dir=video_preprocess_dir,
            batch_size=batch_size
        )

        tracker.stop()
        print("✅ Preprocessing completed.")
        return

    # 4) Load SVM dataset
    dataset = AudioVideoPathDataset(get_project_root(), csv_name=args.csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 5) Load frozen model
    model = ARTModule()
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("✅ Frozen ART model loaded.")

    # 6) Initialize feature processor
    feature_processor = VideoAudioFeatureProcessor(
        video_preprocess_dir=video_preprocess_dir,
        batch_size=batch_size
    )

    # 7) Feature extraction and save .npz
    npz_save_path = os.path.join(args.save_dir, "svm_features.npz")

    def binary_label_fn(label):
        return int(label > 0)

    feature_dataset = FeatureBuilder.encode_and_save_dataset_npz(
        dataloader=dataloader,
        model=model,
        feature_processor=feature_processor,
        output_txt_path=real_output_txt_path,
        binary_label_fn=binary_label_fn,
        save_path_npz=npz_save_path
    )

    # 8) Train SVM
    svm_model, X_test, y_test = SVMTrainer.train(
        dataset=feature_dataset,
        kernel='rbf',
        C=1.0,
        test_size=0.2,
        random_seed=42
    )

    # 9) Evaluate SVM
    plot_dir = os.path.join(args.save_dir, "plots")
    metrics = SVMEvaluator.evaluate(
        model=svm_model,
        X_test=X_test,
        y_test=y_test,
        plot_curves=True,
        plot_dir=plot_dir
    )

    # 10) Print metrics
    print("\n✅ SVM Evaluation Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
