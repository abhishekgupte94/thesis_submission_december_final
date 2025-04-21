import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
from thesis_main_files.models.art_avdf.learning_containers.self_supervised_learning import SelfSupervisedLearning
from thesis_main_files.models.data_loaders.data_loader_ART import VideoAudioFeatureProcessor, VideoAudioDataset


class TrainingPipeline:
    def __init__(self, dataset, batch_size, learning_rate, num_epochs, device, feature_processor):
        self.model = ARTModule().to(device)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.device = device
        self.loss_fn = SelfSupervisedLearning().to(device)
        self.feature_processor = feature_processor

        log_dir = f"runs/ssl_single_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        self.model.train()
        global_step = 0

        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for video_paths, audio_paths, labels in self.dataloader:
                processed_audio_features, processed_video_features = self.feature_processor.create_datasubset(
                    csv_path=None,
                    use_preprocessed=False,
                    video_paths=video_paths,
                    audio_paths=audio_paths
                )

                if processed_audio_features is None or processed_video_features is None:
                    print("Skipping batch due to feature extraction failure.")
                    continue

                audio_features = torch.stack(processed_audio_features).to(self.device)
                video_features = torch.stack(processed_video_features).to(self.device)

                self.optimizer.zero_grad()
                f_art, f_lip = self.model(audio_features, video_features)
                loss = self.loss_fn(f_art, f_lip)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                self.writer.add_scalar("Loss/train", loss.item(), global_step)
                global_step += 1

            avg_loss = running_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

        self.writer.close()
        print("Training completed.")
