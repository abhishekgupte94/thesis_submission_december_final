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
from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass


class TrainingPipeline:
    def __init__(self, dataset, batch_size, learning_rate, num_epochs, device, feature_processor):
        self.model = ARTModule().to(device)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.device = device
        self.loss_fn = SelfSupervisedLearning().to(device)
        self.feature_processor = feature_processor
        self.evaluator = EvaluatorClass()
        log_dir = f"runs/ssl_single_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self,checkpoint_dir):
        self.model.train()
        global_step = 0
        avg_loss = 0
        audio_features = None
        video_features = None
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

                # audio_features = torch.stack(processed_audio_features).to(self.device)
                # video_features = torch.stack(processed_video_features).to(self.device)

                self.optimizer.zero_grad()
                f_art, f_lip = self.model(audio_features = processed_audio_features, video_features= processed_video_features)
                loss,similarity_matrix = self.loss_fn(f_art, f_lip)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                self.writer.add_scalar("Loss/train", loss.item(), global_step)
                global_step += 1
                if (epoch + 1) % 50 == 0:
                    save_path_tsne = os.path.join(checkpoint_dir, f"t_sne_{epoch + 1}.png")
                    save_path_retrieval = os.path.join(checkpoint_dir, f"retrieval_{epoch + 1}.png")

                    self.start_evaluation(self.model, audio_features, video_features,similarity_matrix = similarity_matrix, t_sne_save_path=save_path_tsne,retrieval_save_path=save_path_retrieval)
            avg_loss = running_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
        # ✅ Save checkpoint every 10 epochs
            if (epoch + 1) % 50 == 0:
                save_path = os.path.join(checkpoint_dir, f"art_checkpoint_epoch_{epoch + 1}.pt")
                self.save_state(
                    model=self.model,
                    optimizer=self.optimizer,
                    current_epoch=epoch + 1,
                    current_loss=avg_loss,
                    save_path=save_path
                )
                # self.start_evaluation(self.model,audio_features,video_features)
        # self.save_final_model(self.model)
            # self.start_evaluation(self.model,f_art,f_lip)
        self.writer.close()
        print("Training completed.")
    def save_state(self, model, optimizer, current_epoch, current_loss, save_path="checkpoint_trainer.pt"):
        """
        Save the model and optimizer state (for checkpointing).
        """
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': current_epoch,
            'loss': current_loss
        }, save_path)
        print(f"✅ Training checkpoint saved to: {save_path}")
    def start_evaluation(self, model, audio_inputs, video_inputs, similarity_matrix, t_sne_save_path=None, retrieval_save_path=None):
        """
        Run evaluation using the evaluator.
        """
        self.evaluator.evaluate(model, audio_inputs, video_inputs, t_sne_save_path, retrieval_save_path)
        print("✅ Evaluation complete.")


    # def save_final_model(self, model, save_path="final_trained_model.pt"):
    #     """
    #     Save the final trained model (for deployment or inference).
    #     """
    #     torch.save(model, save_path)
    #     print(f"✅ Final trained model saved to: {save_path}")