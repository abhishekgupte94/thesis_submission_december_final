from thesis_main_files.models.data_loaders.data_loader_ART import VideoAudioFeatureProcessor,VideoAudioDataset
from thesis_main_files.models.art_avdf.learning_containers.self_supervised_learning import SelfSupervisedLearning
from thesis_main_files.models.data_loaders.data_loader_ART import VideoAudioFeatureProcessor,VideoAudioDataset,create_file_paths,get_project_root,convert_paths, get_model_save_paths
from thesis_main_files.main_files.model_training.training_art_wrapper_singleGPU import TrainingPipelineWrapper
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
from thesis_main_files.config import CONFIG
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


import torch

def main():
    # Detect device (GPU or CPU)
    model_save_path, _, _ = get_model_save_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training configuration
    config = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 5,
        "device": device,
        "csv_name": "training_data_two.csv"
    }

    # Instantiate model and wrapper
    model = ARTModule()
    wrapper = TrainingPipelineWrapper(model=model, config=config)

    # Start training
    wrapper.start_training()

    # Save model checkpoint
    # save_path = "checkpoints/artmodel_mac_final.pt"
    print(f"Saving model to: {model_save_path}")
    wrapper.save_final_model(model=model, save_path=model_save_path)

    # wrapper.save_state(
    #     model=model,
    #     optimizer=wrapper.pipeline.optimizer,
    #     current_epoch=config["num_epochs"],
    #     current_loss=0.0  # Or log the final loss from training loop if available
    # )
    # print("✅ Model saved.")

if __name__ == "__main__":
    main()

