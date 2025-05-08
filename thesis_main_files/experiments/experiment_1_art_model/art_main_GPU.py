from thesis_main_files.main_files.model_training.training_art_wrapper_GPU import TrainingPipelineWrapper
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def main(rank, world_size):
    # -------------------- INIT --------------------
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    config = {
        "csv_name": "training_data_two.csv",
        "batch_size": 128,
        "learning_rate": 1e-4,
        "num_epochs": 30,
        "device": torch.device(f"cuda:{rank}")
    }

    # -------------------- TRAIN --------------------
    wrapper = TrainingPipelineWrapper(config=config, rank=rank, world_size=world_size)
    wrapper.start_training()

    # -------------------- SAVE FINAL MODEL (only on rank 0) --------------------
    if rank == 0:
        print("🧠 Saving final trained model...")
        model = wrapper.pipeline.model
        os.makedirs("checkpoints", exist_ok=True)

        wrapper.save_final_model(
            model=model.module if hasattr(model, "module") else model,
            save_path="checkpoints/artmodel_ddp_final.pt"
        )

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
