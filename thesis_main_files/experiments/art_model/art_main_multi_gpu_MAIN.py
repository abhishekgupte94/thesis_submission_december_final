# Ensure the root of the project is in sys.path
import os
import sys
# import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from codecarbon import EmissionsTracker

# import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
from thesis_main_files.models.data_loaders.data_loader_ART import VideoAudioDataset, VideoAudioFeatureProcessor, convert_paths, preprocess_videos_before_training, get_project_root
from thesis_main_files.models.art_avdf.training_pipeline.training_ART_Multi_GPU import TrainingPipeline

def main():
    # 1) Argument parser
    parser = argparse.ArgumentParser(description="Train ART model or preprocess videos")
    parser.add_argument('--preprocess', action='store_true', help='Only preprocess videos and exit')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU or per video batch in preprocessing')
    parser.add_argument('--csv_file', type=str, default="training_data_two.csv", help='CSV filename for training or preprocessing')
    args = parser.parse_args()

    # 2) Project paths
    os.makedirs("checkpoint/", exist_ok=True)
    os.makedirs("save_final_model/", exist_ok=True)
    os.makedirs("carbon_logs_preprocessing", exist_ok=True)

    # 3) Select correct path function based on CSV
    if args.csv_file == "training_data_two.csv":
        csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path = convert_paths()
    elif args.csv_file == "training_data_svm_final.csv":
        from thesis_main_files.models.data_loaders.data_loader_ART import convert_paths_for_svm_train_preprocess
        csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path = convert_paths_for_svm_train_preprocess()
    elif args.csv_file == "val_data_for_svm.csv":
        from thesis_main_files.models.data_loaders.data_loader_ART import convert_paths_for_svm_val_preprocess
        csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path = convert_paths_for_svm_val_preprocess()
    else:
        raise ValueError(f"Unsupported CSV file '{args.csv_file}'. Please use one of: training_data_two.csv, training_data_svm_final.csv, val_data_for_svm.csv.")

    batch_size = args.batch_size

    if args.preprocess:
        tracker = EmissionsTracker(
            project_name="preprocessing",
            output_dir="carbon_logs_preprocessing",
        )
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

    # DDP setup
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    # 4) Load dataset with custom CSV name
    dataset = VideoAudioDataset(get_project_root(), csv_name=args.csv_file)

    # 5) Feature processor
    feature_processor = VideoAudioFeatureProcessor(
        video_preprocess_dir=video_preprocess_dir,
        batch_size=batch_size
    )

    # 6) Initialize trainer
    trainer = TrainingPipeline(
        dataset=dataset,
        batch_size=batch_size,
        learning_rate=1e-4,
        num_epochs=150,
        device=torch.device(f"cuda:{local_rank}"),
        feature_processor=feature_processor,
        output_txt_path=real_output_txt_path,
        local_rank=local_rank
    )

    # 7) Start training
    trainer.train("checkpoint/")
    trainer.save_final_state("save_final_model/final_model.pt")

    # 8) Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()