import torch
from thesis_main_files.models.art_avdf.training_pipeline.training_ART_GPU import TrainingPipeline
from thesis_main_files.models.data_loaders.data_loader_ART import (
    VideoAudioFeatureProcessor,
    VideoAudioDataset,
    create_file_paths,
    get_project_root,
    convert_paths
)

from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass


class TrainingPipelineWrapper:
    def __init__(self, config=None, rank=0, world_size=1):
        """
        Initializes and prepares the training pipeline for DDP training.

        Args:
            config (dict, optional): Configuration dictionary. Keys can include:
                - batch_size (int)
                - learning_rate (float)
                - num_epochs (int)
                - csv_name (str)
            rank (int): Current GPU/process rank for DDP.
            world_size (int): Total number of processes (GPUs).
        """
        if config is None:
            config = {}

        self.rank = rank
        self.world_size = world_size

        # Load paths from utilities
        (csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin,
         video_preprocess_dir, feature_dir_vid, audio_dir, video_dir,
         real_output_txt_path) = convert_paths()

        # Create dataset file paths
        csv_name = config.get("csv_name", "training_data_two.csv")
        video_paths, audio_paths, labels = create_file_paths(get_project_root(), csv_name)

        # Prepare the feature processor
        batch_size = config.get("batch_size", 128)
        processor = VideoAudioFeatureProcessor(
            video_preprocess_dir=video_preprocess_dir,
            audio_preprocess_dir=audio_preprocess_dir,
            feature_dir_vid=feature_dir_vid,
            feature_dir_audio=feature_dir_audio,
            batch_size=batch_size,
            video_save_dir=video_preprocess_dir,
            output_txt_file=real_output_txt_path
        )

        # Prepare the dataset
        dataset = VideoAudioDataset(
            project_dir_curr=get_project_root(),
            csv_name=csv_name
        )

        # Training params
        learning_rate = config.get("learning_rate", 1e-4)
        num_epochs = config.get("num_epochs", 10)
        device = torch.device(f"cuda:{rank}")

        # Evaluator (optional for post-training)
        self.evaluator = EvaluatorClass()

        # Initialize DDP training pipeline
        self.pipeline = TrainingPipeline(
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=device,
            feature_processor=processor,
            rank=rank,
            world_size=world_size
        )

    def start_training(self):
        print(f"[GPU {self.rank}] Starting training...")
        self.pipeline.train()
        if self.rank == 0:
            print("Training complete.")

    def save_state(self, model, optimizer, current_epoch, current_loss, save_path="checkpoint_trainer.pt"):
        """
        Save the model and optimizer state (for checkpointing).
        Only rank 0 saves to avoid duplicate writes.
        """
        if self.rank == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': current_epoch,
                'loss': current_loss
            }, save_path)
            print(f"✅ Training checkpoint saved to: {save_path}")

    def start_evaluation(self, model, audio_inputs, video_inputs):
        """
        Runs evaluation only on rank 0.
        """
        if self.rank == 0:
            self.evaluator.evaluate(model, audio_inputs, video_inputs)
            print("✅ Evaluation complete.")

    def save_final_model(self, model, save_path="final_trained_model.pt"):
        """
        Save the final trained model (for deployment or inference).
        Unwraps model.module if wrapped in DDP.
        Only rank 0 saves.
        """
        if self.rank == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save, save_path)
            print(f"✅ Final trained model saved to: {save_path}")
