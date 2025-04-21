from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
from thesis_main_files.models.art_avdf.learning_containers.self_supervised_learning import SelfSupervisedLearning
from thesis_main_files.models.data_loaders.data_loader_ART import VideoAudioFeatureProcessor, VideoAudioDataset, create_file_paths, get_project_root, convert_paths


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


class TrainingPipeline:
    def __init__(self, dataset, batch_size, learning_rate, num_epochs, device, feature_processor, rank, world_size):
        self.rank = rank
        self.world_size = world_size

        self.model = ARTModule().to(device)
        self.model = DDP(self.model, device_ids=[rank])

        self.dataset = dataset
        self.sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, sampler=self.sampler, num_workers=4, pin_memory=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.device = device
        self.loss_fn = SelfSupervisedLearning().to(device)
        self.feature_processor = feature_processor

        if rank == 0:
            log_dir = f"runs/ssl_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def train(self):
        self.model.train()
        global_step = 0

        for epoch in range(self.num_epochs):
            self.sampler.set_epoch(epoch)  # Enables proper shuffling
            running_loss = 0.0

            for video_paths, audio_paths, labels in self.dataloader:
                processed_audio_features, processed_video_features = self.feature_processor.create_datasubset(
                    csv_path=None,
                    use_preprocessed=False,
                    video_paths=video_paths,
                    audio_paths=audio_paths
                )

                if processed_audio_features is None or processed_video_features is None:
                    if self.rank == 0:
                        print("Skipping batch due to feature extraction failure.")
                    continue

                audio_features = torch.stack(processed_audio_features).to(self.device)
                video_features = torch.stack(processed_video_features).to(self.device)

                self.optimizer.zero_grad()
                f_art, f_lip = self.model(audio_features, video_features)
                loss = self.loss_fn(f_art, f_lip)

                # Reduce loss across all GPUs
                reduced_loss = reduce_tensor(loss, self.world_size)

                loss.backward()
                self.optimizer.step()

                if self.rank == 0:
                    running_loss += reduced_loss.item()
                    self.writer.add_scalar("Loss/train", reduced_loss.item(), global_step)
                    global_step += 1

            if self.rank == 0:
                avg_loss = running_loss / len(self.dataloader)
                print(f"[GPU {self.rank}] Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

        if self.rank == 0:
            self.writer.close()
            print("Training completed.")
