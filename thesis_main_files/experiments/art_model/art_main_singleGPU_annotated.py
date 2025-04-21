from thesis_main_files.main_files.model_training.training_art_wrapper_singleGPU import TrainingPipelineWrapper
import os
import torch

def main():
    # -------------------- CONFIG --------------------
    config = {
        "csv_name": "training_data_two.csv",
        "batch_size": 128,
        "learning_rate": 1e-4,
        "num_epochs": 30,
    }

    # -------------------- INIT PIPELINE --------------------
    wrapper = TrainingPipelineWrapper(config=config)

    # -------------------- TRAIN --------------------
    wrapper.start_training()

    # -------------------- SAVE FINAL MODEL --------------------
    print("🧠 Saving final trained model...")
    model = wrapper.pipeline.model
    os.makedirs("checkpoints", exist_ok=True)
    wrapper.save_final_model(model=model, save_path="checkpoints/artmodel_final.pt")


if __name__ == "__main__":
    main()
