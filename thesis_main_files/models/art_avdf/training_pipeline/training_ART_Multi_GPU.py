# import os
# from datetime import datetime
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.distributed as dist
# from torch.utils.data import DataLoader, DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
#
# log_dir = f"runs/ssl_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# # self.writer = SummaryWriter(log_dir=log_dir) if dist.get_rank() == 0 else None
#
# from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
# from thesis_main_files.models.art_avdf.learning_containers.self_supervised_learning import SelfSupervisedLearning
# from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass
# from thesis_main_files.utils.files_imp import create_manifest_from_selected_files
# # from thesis_main_files.models.data_loaders.data_loader_ART import create_manifest_from_selected_files
# # from pyJoules.energy_meter import EnergyMeter
# # from pyJoules.handler.csv_handler import CSVHandler
# from thesis_main_files.models.art_avdf.encoders.new_encoder_for_ssl.complete_pipeline_cpu import CPUOptimizedAlignment
# from thesis_main_files.models.art_avdf.encoders.new_encoder_for_ssl.alignment_pipeline_gpu import GPUOptimizedAlignment, SelfSupervisedAVLoss
# log_dir = f"runs/ssl_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#
# class TrainingPipeline:
#     def __init__(self, dataset, batch_size, learning_rate, num_epochs, device, feature_processor, output_txt_path, local_rank):
#         self.local_rank = local_rank
#         self.device = torch.device(f'cuda:{local_rank}')
#         torch.cuda.set_device(self.device)
#         log_dir = f"runs/ssl_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#
#         self.writer = SummaryWriter(log_dir=log_dir) if dist.get_rank() == 0 else None
#
#         # 🔄 CHANGE: use GPUOptimizedAlignment instead of ARTModule
#         self.model = GPUOptimizedAlignment(
#             rank=local_rank,
#             world_size=dist.get_world_size(),
#             K=50,
#             common_dim=512
#         ).to(self.device)
#         self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
#
#         self.dataset = dataset
#         self.sampler = DistributedSampler(self.dataset)
#
#         self.dataloader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             sampler=self.sampler,
#             num_workers=8,
#             pin_memory=True,
#             persistent_workers=True
#         )
#
#         self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
#         self.num_epochs = num_epochs
#
#         # 🔄 CHANGE: use SelfSupervisedAVLoss instead of SelfSupervisedLearning
#         self.loss_fn = SelfSupervisedAVLoss(initial_temperature=0.1).to(self.device)
#
#         self.feature_processor = feature_processor
#         # self.evaluator = EvaluatorClass()  # 🔒 Commented out for now
#         self.output_txt_path = output_txt_path
#
#         log_dir = f"runs/ssl_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         self.writer = SummaryWriter(log_dir=log_dir)
#
#     # def extract_features(self, video_paths):
#     #     processed_audio_features, processed_video_features = self.feature_processor.create_datasubset(
#     #         csv_path=None,
#     #         use_preprocessed=False,
#     #         video_paths=video_paths
#     #     )
#     #     return processed_audio_features, processed_video_features
#
#     def extract_features(self, video_paths):
#         """
#         ENHANCED: Better error handling and device management for cross-GPU feature extraction
#         """
#         processed_audio_features, processed_video_features = self.feature_processor.create_datasubset(
#             csv_path=None,
#             use_preprocessed=False,
#             video_paths=video_paths
#         )
#
#         # Move features to training device if extraction succeeded
#         if processed_audio_features is not None and processed_video_features is not None:
#             # Features come back on CPU, move to training device
#             processed_audio_features = processed_audio_features.to(self.device, non_blocking=True)
#             processed_video_features = processed_video_features.to(self.device, non_blocking=True)
#
#         return processed_audio_features, processed_video_features
#
#
#     def train(self, checkpoint_dir):
#         self.model.train()
#         global_step = 0
#         avg_loss = 0
#
#         for epoch in range(self.num_epochs):
#             self.sampler.set_epoch(epoch)
#             running_loss = 0.0
#
#             # for batch in self.dataloader:
#             #     video_paths = batch["video_path"]  # List[str]
#             #     labels = batch["label"]  # List[int] or tensor
#             #     processed_audio_features, processed_video_features = self.extract_features(video_paths)
#             #     # after extraction, force both onto the model’s device
#             #
#             #     if processed_audio_features is None or processed_video_features is None:
#             #         print("Skipping batch due to feature extraction failure.")
#             #         continue
#             #     processed_audio_features = processed_audio_features.to(self.device, non_blocking=True)
#             #     processed_video_features = processed_video_features.to(self.device, non_blocking=True)
#             #
#             #     self.optimizer.zero_grad()
#             #
#             #     # 🔄 CHANGE: GPUOptimizedAlignment outputs a dict
#             #     output = self.model(
#             #         audio_features=processed_audio_features,
#             #         video_features=processed_video_features
#             #     )
#             #
#             #     # Global pooled features for loss
#             #     audio_global = output['audio_aligned'].mean(dim=1)  # [B, 512]
#             #     video_global = output['video_aligned'].mean(dim=1)  # [B, 512]
#             #
#             #     # 🔄 CHANGE: use SelfSupervisedAVLoss
#             #     loss = self.loss_fn(audio_global, video_global)
#             #
#             #     loss.backward()
#             #     self.optimizer.step()
#             #     if self.writer is not None:
#             #         self.writer.add_scalar("train/loss_step", loss.item(), global_step)
#             #         for i, pg in enumerate(self.optimizer.param_groups):
#             #             self.writer.add_scalar(f"train/lr_group_{i}", pg["lr"], global_step)
#             #
#             #     running_loss += loss.item()
#             #
#             #     if dist.get_rank() == 0:
#             #         self.writer.add_scalar("Loss/train", loss.item(), global_step)
#             #
#             #     global_step += 1
#             ### NEW GPU LOGIC
#             for batch_idx, batch in enumerate(self.dataloader):
#                 video_paths = batch["video_path"]
#                 labels = batch["label"]
#
#                 processed_audio_features, processed_video_features = self.extract_features(video_paths)
#
#                 if processed_audio_features is None or processed_video_features is None:
#                     if dist.get_rank() == 0:
#                         print(f"⏭️  Skipping batch {batch_idx} due to feature extraction failure.")
#                     continue
#
#                 self.optimizer.zero_grad()
#
#                 output = self.model(
#                     audio_features=processed_audio_features,
#                     video_features=processed_video_features
#                 )
#
#                 # Global pooled features for loss
#                 audio_global = output['audio_aligned'].mean(dim=1)
#                 video_global = output['video_aligned'].mean(dim=1)
#
#                 loss = self.loss_fn(audio_global, video_global)
#                 loss.backward()
#                 self.optimizer.step()
#
#                 running_loss += loss.item()
#
#                 if self.writer is not None:
#                     self.writer.add_scalar("train/loss_step", loss.item(), global_step)
#                     for i, pg in enumerate(self.optimizer.param_groups):
#                         self.writer.add_scalar(f"train/lr_group_{i}", pg["lr"], global_step)
#
#                 global_step += 1
#
#                 # Periodic performance reporting
#                 if batch_idx % 50 == 0 and dist.get_rank() == 0:
#                     print(f"📊 Batch {batch_idx}: Loss={loss.item():.4f}")
#                 # 🔒 Commented out evaluator for now
#                 # if (epoch + 1) % 50 == 0 and dist.get_rank() == 0:
#                 #     save_path_tsne = os.path.join(checkpoint_dir, f"t_sne_{epoch + 1}.png")
#                 #     save_path_retrieval = os.path.join(checkpoint_dir, f"retrieval_{epoch + 1}.png")
#                 #     self.start_evaluation(self.model.module, None, None, similarity_matrix, save_path_tsne, save_path_retrieval)
#
#             avg_loss = running_loss / len(self.dataloader)
#             print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
#             if self.writer is not None:
#                 self.writer.add_scalar("train/loss_epoch", avg_loss, epoch + 1)
#
#             if (epoch + 1) % 50 == 0 and dist.get_rank() == 0:
#                 save_path = os.path.join(checkpoint_dir, f"art/_checkpoint_epoch_{epoch + 1}.pt")
#                 self.save_state(
#                     model=self.model.module,
#                     optimizer=self.optimizer,
#                     current_epoch=epoch + 1,
#                     current_loss=avg_loss,
#                     save_path=save_path
#                 )
#
#         if dist.get_rank() == 0:
#             self.writer.close()
#         print(f"Training completed on rank {dist.get_rank()}.")
#
#     def save_state(self, model, optimizer, current_epoch, current_loss, save_path="checkpoint_trainer.pt"):
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'epoch': current_epoch,
#             'loss': current_loss
#         }, save_path)
#         print(f"✅ Training checkpoint saved to: {save_path}")
#
#     def save_final_state(self, path="final_model.pt"):
#         if dist.get_rank() == 0:
#             torch.save({
#                 'model_state_dict': self.model.module.state_dict(),
#                 'optimizer_state_dict': self.optimizer.state_dict(),
#                 'epoch': self.num_epochs
#             }, path)
#             print(f"✅ Final model saved at: {path}")

import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

log_dir = f"runs/ssl_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# self.writer = SummaryWriter(log_dir=log_dir) if dist.get_rank() == 0 else None

from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
from thesis_main_files.models.art_avdf.learning_containers.self_supervised_learning import SelfSupervisedLearning
from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass
from thesis_main_files.utils.files_imp import create_manifest_from_selected_files
# from thesis_main_files.models.data_loaders.data_loader_ART import create_manifest_from_selected_files
# from pyJoules.energy_meter import EnergyMeter
# from pyJoules.handler.csv_handler import CSVHandler
from thesis_main_files.models.art_avdf.encoders.new_encoder_for_ssl.complete_pipeline_cpu import CPUOptimizedAlignment
from thesis_main_files.models.art_avdf.encoders.new_encoder_for_ssl.alignment_pipeline_gpu import GPUOptimizedAlignment, SelfSupervisedAVLoss
log_dir = f"runs/ssl_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class TrainingPipeline:
    def __init__(self, dataset, batch_size, learning_rate, num_epochs, device, feature_processor, output_txt_path, local_rank):
        if dist.is_initialized():
            ra = dist.get_rank()
            try:
                mvit_dev = feature_processor.feature_extractor.mvit_adapter.device
                ast_dev = feature_processor.feature_extractor.audio_extractor.device
            except Exception:
                mvit_dev = ast_dev = "unknown"
            print(f"[rank {ra}] trainer={self.device} | mvit={mvit_dev} | ast={ast_dev}")

        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        log_dir = f"runs/ssl_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.writer = SummaryWriter(log_dir=log_dir) if dist.get_rank() == 0 else None

        # ðŸ”„ CHANGE: use GPUOptimizedAlignment instead of ARTModule
        self.model = GPUOptimizedAlignment(
            rank=local_rank,
            world_size=dist.get_world_size(),
            K=50,
            common_dim=512
        ).to(self.device)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank],output_device=local_rank, find_unused_parameters=False)

        self.dataset = dataset
        self.sampler = DistributedSampler(self.dataset)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=min(8, (os.cpu_count() or 8)),
            pin_memory=True,
            prefetch_factor=4,  # ← keep workers ahead
            persistent_workers=False,  # ← avoid first-batch stalls during bring-up
            drop_last=True,  # ← cleaner DDP sync
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

        # ðŸ”„ CHANGE: use SelfSupervisedAVLoss instead of SelfSupervisedLearning
        self.loss_fn = SelfSupervisedAVLoss(initial_temperature=0.1).to(self.device)

        self.feature_processor = feature_processor
        # self.evaluator = EvaluatorClass()  # ðŸ”’ Commented out for now
        self.output_txt_path = output_txt_path

        log_dir = f"runs/ssl_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def extract_features(self, video_paths):
        processed_audio_features, processed_video_features = self.feature_processor.create_datasubset(
            csv_path=None,
            use_preprocessed=False,
            video_paths=video_paths
        )
        return processed_audio_features, processed_video_features

    def train(self, checkpoint_dir):
        self.model.train()
        global_step = 0
        avg_loss = 0

        for epoch in range(self.num_epochs):
            self.sampler.set_epoch(epoch)
            running_loss = 0.0

            for batch in self.dataloader:
                video_paths = batch["video_path"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                processed_audio_features, processed_video_features = self.extract_features(video_paths)
                # after extraction, force both onto the modelâ€™s device

                if processed_audio_features is None or processed_video_features is None:
                    print("Skipping batch due to feature extraction failure.")
                    continue
                processed_audio_features = processed_audio_features.to(self.device, non_blocking=True)
                processed_video_features = processed_video_features.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                # ðŸ”„ CHANGE: GPUOptimizedAlignment outputs a dict
                output = self.model(
                    audio_features=processed_audio_features,
                    video_features=processed_video_features
                )

                # Global pooled features for loss
                audio_global = output['audio_aligned'].mean(dim=1)  # [B, 512]
                video_global = output['video_aligned'].mean(dim=1)  # [B, 512]

                # ðŸ”„ CHANGE: use SelfSupervisedAVLoss
                loss = self.loss_fn(audio_global, video_global)

                loss.backward()
                self.optimizer.step()
                if self.writer is not None:
                    self.writer.add_scalar("train/loss_step", loss.item(), global_step)
                    for i, pg in enumerate(self.optimizer.param_groups):
                        self.writer.add_scalar(f"train/lr_group_{i}", pg["lr"], global_step)

                running_loss += loss.item()

                if dist.get_rank() == 0:
                    self.writer.add_scalar("Loss/train", loss.item(), global_step)

                global_step += 1

                # ðŸ”’ Commented out evaluator for now
                # if (epoch + 1) % 50 == 0 and dist.get_rank() == 0:
                #     save_path_tsne = os.path.join(checkpoint_dir, f"t_sne_{epoch + 1}.png")
                #     save_path_retrieval = os.path.join(checkpoint_dir, f"retrieval_{epoch + 1}.png")
                #     self.start_evaluation(self.model.module, None, None, similarity_matrix, save_path_tsne, save_path_retrieval)

            avg_loss = running_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
            if self.writer is not None:
                self.writer.add_scalar("train/loss_epoch", avg_loss, epoch + 1)

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
        print(f"âœ… Training checkpoint saved to: {save_path}")

    def save_final_state(self, path="final_model.pt"):
        if dist.get_rank() == 0:
            torch.save({
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.num_epochs
            }, path)
            print(f"âœ… Final model saved at: {path}")
