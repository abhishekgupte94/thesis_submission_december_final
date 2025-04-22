# Ensure the root of the project is in sys.path
import os
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



from thesis_main_files.main_files.model_training.training_art_wrapper_singleGPU import TrainingPipelineWrapper
# import os
import torch



def main():
    # -------------------- CONFIG --------------------
    config = {
        "csv_name": "training_data_two.csv",
        "batch_size": 128,
        "learning_rate": 1e-4,
        "num_epochs": 30,
    }
    checkpoint_dir = "/content/drive/MyDrive/checkpoints"
    # -------------------- INIT PIPELINE --------------------
    wrapper = TrainingPipelineWrapper(config=config)

    # -------------------- TRAIN --------------------
    wrapper.start_training(checkpoint_dir)

    # -------------------- SAVE FINAL MODEL --------------------
    print("🧠 Saving final trained model...")
    model = wrapper.pipeline.model
    os.makedirs("checkpoints", exist_ok=True)
    wrapper.save_final_model(model=model, save_path="checkpoints/artmodel_final.pt")


if __name__ == "__main__":
    main()
