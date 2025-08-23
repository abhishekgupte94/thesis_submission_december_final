# import torch
# from thesis_main_files.models.art_avdf.training_pipeline.training_ART_test import TestTrainingPipelineWrapper
# # from models.art_avdf.training_pipeline.training_ART_test import TrainingPipeline
# # from models.art_avdf.training_pipeline.training_ART_test import TestTrainingPipelineWrapper
# from thesis_main_files.models.data_loaders.data_loader_ART import (
#     VideoAudioFeatureProcessor, VideoAudioDataset,
#     create_file_paths, get_project_root, convert_paths
# )
# from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass
# from thesis_main_files.models.art_avdf.learning_containers.self_supervised_learning import SelfSupervisedLearning
# from thesis_main_files.config import CONFIG
# import os
#
# # =======================================
# # ✅ NEW: TEST-TIME TRAINING WRAPPER (DIRECT INPUTS)
# # =======================================
#
#
# class TestTrainingPipelineWrapper:
#     def __init__(self, model, audio_batches, video_batches, config=None):
#         """
#         Args:
#             model (nn.Module): The model to be trained.
#             audio_batches (List[Tensor]): List of audio feature tensors [B, D]
#             video_batches (List[Tensor]): List of video feature tensors [B, D]
#             config (dict): Training config, supports:
#                 - device, learning_rate, num_epochs
#         """
#         if config is None:
#             config = {}
#
#         assert len(audio_batches) == len(video_batches), "Mismatch in number of audio/video batches."
#
#         self.model = model.to(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#         self.device = next(self.model.parameters()).device
#
#         self.audio_batches = [a.to(self.device) for a in audio_batches]
#         self.video_batches = [v.to(self.device) for v in video_batches]
#
#         self.learning_rate = config.get("learning_rate", 1e-4)
#         self.num_epochs = config.get("num_epochs", 100)
#
#         self.loss_fn = SelfSupervisedLearning().to(self.device)
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
#         self.evaluator = EvaluatorClass()
#
#     def start_training(self,checkpoint_dir):
#         print("🔁 Starting batch-based training...")
#         self.model.train()
#         for epoch in range(self.num_epochs):
#             epoch_loss = 0.0
#             for i, (audio, video) in enumerate(zip(self.audio_batches, self.video_batches)):
#                 self.optimizer.zero_grad()
#
#                 f_art, f_lip = self.model(audio, video)
#                 loss = self.loss_fn(f_art, f_lip)
#
#                 loss.backward()
#                 self.optimizer.step()
#                 epoch_loss += loss.item()
#                 avg_loss = running_loss / len(self.dataloader)
#                 print(f"  [Epoch {epoch+1}/{self.num_epochs}] Batch {i+1} Loss: {loss.item():.4f}")
#                 if (epoch + 1) % 100 == 0:
#                     save_path = os.path.join(checkpoint_dir, f"art_checkpoint_epoch_{epoch + 1}.pt")
#                     self.save_state(
#                         model=self.model,
#                         optimizer=self.optimizer,
#                         current_epoch=epoch + 1,
#                         current_loss=avg_loss,
#                         save_path=save_path
#                     )
#                 print(f"✅ Epoch {epoch+1} Avg Loss: {epoch_loss / len(self.audio_batches):.4f}")
#
#     def start_evaluation(self):
#         print("📊 Running batch-wise evaluation...")
#         self.model.eval()
#         with torch.no_grad():
#             for i, (audio, video) in enumerate(zip(self.audio_batches, self.video_batches)):
#                 print(f"  [Eval Batch {i+1}]")
#                 self.evaluator.evaluate(
#                     model=self.model,
#                     audio_inputs=audio,
#                     video_inputs=video
#                 )
#
#     def save_state(self, current_epoch=0, current_loss=0.0):
#         print("💾 Saving model checkpoint...")
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'epoch': current_epoch,
#             'loss': current_loss
#         }, 'checkpoint_test_trainer.pt')
#     # def start_training(self,checkpoint_dir):
#     #     print("Starting training on single GPU...")
#     #     self.pipeline.train(checkpoint_dir)
#     #     print("✅ Training complete.")
#     #
#     #
#     #
#     # def save_final_model(self, model, save_path="final_trained_model.pt"):
#     #     """
#     #     Save the final trained model (for deployment or inference).
#     #     """
#     #     torch.save(model, save_path)
#     #     print(f"✅ Final trained model saved to: {save_path}")