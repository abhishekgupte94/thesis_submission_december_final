import os
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU or per video batch in preprocessing')
    args = parser.parse_args()

    # 2) Predefined project paths (already configured internally)
    # project_dir = "path/to/project"/
    # video_preprocess_dir = "path/to/video_preprocess"
    # feature_dir_vid = "path/to/feature_dir"
    # output_txt_path = "path/to/output_txt"
    # checkpoint_dir = "checkpoints"
    # save_final_state_dir = "save_final_state"

    # 3) Create necessary directories
    os.makedirs("checkpoint/", exist_ok=True)
    os.makedirs("save_final_model/", exist_ok=True)
    csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path = convert_paths()
    batch_size = 64
    if args.preprocess:
        preprocess_videos_before_training(
            csv_path=csv_path,
            csv_column="video_file",
            output_dir=video_preprocess_dir,
            batch_size=batch_size  # assuming default batch size
        )
        print("✅ Preprocessing completed.")
        return

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # 2) Init process group
    dist.init_process_group(backend='nccl')

    # 3) Project root and paths (customize as needed)
    # project_dir = "path/to/project"  # TODO: Set this
    # output_txt_path = "path/to/lip_manifest.txt"  # TODO: Set this
    # checkpoint_dir = "checkpoints/"  # TODO: Set this
    # 4) Load dataset

    dataset = VideoAudioDataset(get_project_root())

    # 5) Create feature processor
    # video_preprocess_dir = "path/to/swin_input_dir"  # TODO
    # feature_dir_vid = "path/to/feature_output_dir"   # TODO
    # video_save_dir = video_preprocess_dir  # can reuse
    # batch_size = 128
    feature_processor = VideoAudioFeatureProcessor(
        video_preprocess_dir=video_preprocess_dir,
        # feature_dir_vid=feature_dir_vid,
        batch_size=batch_size
        # video_save_dir=video_preprocess_dir,
        # output_txt_file=real_output_txt_path
    )
    num_epochs = 150
    lr = 1e-4
    # 6) Initialize training pipeline
    trainer = TrainingPipeline(
        dataset=dataset,
        batch_size=batch_size,
        learning_rate=lr,
        num_epochs=num_epochs,
        device=torch.device(f"cuda:{local_rank}"),
        feature_processor=feature_processor,
        output_txt_path=real_output_txt_path,
        local_rank=local_rank
    )
    # 7) Start trainingw
    trainer.train("checkpoints/")
    trainer.save_final_state("save_final_state/final_model.pt")

    # 8) Cleanup DDP
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
