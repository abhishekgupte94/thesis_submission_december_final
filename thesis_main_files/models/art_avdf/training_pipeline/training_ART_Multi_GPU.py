import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
from thesis_main_files.models.art_avdf.learning_containers.self_supervised_learning import SelfSupervisedLearning
from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass
# from thesis_main_files.models.data_loaders.data_loader_ART import create_manifest_from_selected_files

def create_manifest_from_selected_files(selected_video_paths, output_txt_path):
    """
    Writes a text file for selected .mp4 videos with format:
    filename.mp4 0

    Args:
        selected_video_paths (list of str or Path): List of selected video file paths.
        output_txt_path (str or Path): Where to save the manifest text file.
    """
    from pathlib import Path

    output_txt_path = Path(output_txt_path)

    with output_txt_path.open("w") as f:
        for video_path in selected_video_paths:
            video_file = Path(video_path).name  # extract only the filename
            f.write(f"{video_file} 0\n")

    print(f"✅ Manifest created at: {output_txt_path} ({len(selected_video_paths)} entries)")

class TrainingPipeline:
    def __init__(self, dataset, batch_size, learning_rate, num_epochs, device, feature_processor, output_txt_path, local_rank):
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)

        self.model = ARTModule().to(self.device)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])

        self.dataset = dataset
        self.sampler = DistributedSampler(self.dataset)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.loss_fn = SelfSupervisedLearning().to(self.device)
        self.feature_processor = feature_processor
        self.evaluator = EvaluatorClass()
        self.output_txt_path = output_txt_path

        log_dir = f"runs/ssl_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, checkpoint_dir):
        self.model.train()
        global_step = 0
        avg_loss = 0

        for epoch in range(self.num_epochs):
            self.sampler.set_epoch(epoch)

            running_loss = 0.0

            for video_paths, audio_paths, labels in self.dataloader:
                create_manifest_from_selected_files(video_paths, self.output_txt_path)

                processed_audio_features, processed_video_features = self.feature_processor.create_datasubset(
                    csv_path=None,
                    use_preprocessed=False,
                    video_paths=video_paths
                )

                if processed_audio_features is None or processed_video_features is None:
                    print("Skipping batch due to feature extraction failure.")
                    continue

                self.optimizer.zero_grad()

                f_art, f_lip = self.model(
                    audio_features=processed_audio_features,
                    video_features=processed_video_features
                )
                loss, similarity_matrix = self.loss_fn(f_art, f_lip)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if dist.get_rank() == 0:
                    self.writer.add_scalar("Loss/train", loss.item(), global_step)

                global_step += 1

                if (epoch + 1) % 50 == 0 and dist.get_rank() == 0:
                    save_path_tsne = os.path.join(checkpoint_dir, f"t_sne_{epoch + 1}.png")
                    save_path_retrieval = os.path.join(checkpoint_dir, f"retrieval_{epoch + 1}.png")
                    self.start_evaluation(self.model.module, None, None, similarity_matrix, save_path_tsne, save_path_retrieval)

            avg_loss = running_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

            if (epoch + 1) % 50 == 0 and dist.get_rank() == 0:
                save_path = os.path.join(checkpoint_dir, f"art_checkpoint_epoch_{epoch + 1}.pt")
                self.save_state(
                    model=self.model.module,
                    optimizer=self.optimizer,
                    current_epoch=epoch + 1,
                    current_loss=avg_loss,
                    save_path=save_path
                )

        if dist.get_rank() == 0:
            self.writer.close()

        print(f"Training completed on rank {dist.get_rank()}.")

    def save_state(self, model, optimizer, current_epoch, current_loss, save_path="checkpoint_trainer.pt"):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': current_epoch,
            'loss': current_loss
        }, save_path)
        print(f"✅ Training checkpoint saved to: {save_path}")
    def save_final_state(self, path="final_model.pt"):
        if dist.get_rank() == 0:
            torch.save({
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.num_epochs
            }, path)
            print(f"✅ Final model saved at: {path}")
    def start_evaluation(self, model, audio_inputs, video_inputs, similarity_matrix, t_sne_save_path=None, retrieval_save_path=None):
        self.evaluator.evaluate(model, audio_inputs, video_inputs, t_sne_save_path, retrieval_save_path)
        print("✅ Evaluation complete.")
