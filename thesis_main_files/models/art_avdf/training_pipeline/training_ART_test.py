from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from thesis_main_files.models.data_loaders.data_loader_ART import VideoAudioFeatureProcessor,VideoAudioDataset
from thesis_main_files.models.art_avdf.learning_containers.self_supervised_learning import SelfSupervisedLearning
from thesis_main_files.models.data_loaders.data_loader_ART import VideoAudioFeatureProcessor,VideoAudioDataset,create_file_paths,get_project_root,convert_paths
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin, video_preprocess_dir, feature_dir_vid, audio_dir, video_dir, real_output_txt_path = convert_paths()
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
# Step 2: Create file paths for video/audio + labels
# video_paths, audio_paths, labels = create_file_paths(get_project_root(), "training_data_two.csv")

# Step 3: Initialize the Processor with needed dirs and batch_size
batch_size = 32
import os
import torch
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass
from thesis_main_files.models.art_avdf.learning_containers.self_supervised_learning import SelfSupervisedLearning


class TestTrainingPipelineWrapper:
    def __init__(self, model, audio_batches, video_batches, config=None):
        """
        Args:
            model (nn.Module): The model to be trained.
            audio_batches (List[Tensor]): List of audio feature tensors [B, D]
            video_batches (List[Tensor]): List of video feature tensors [B, D]
            config (dict): Training config, supports:
                - device, learning_rate, num_epochs
        """
        if config is None:
            config = {}

        assert len(audio_batches) == len(video_batches), "Mismatch in number of audio/video batches."

        self.model = model.to(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = next(self.model.parameters()).device

        self.audio_batches = [a.to(self.device) for a in audio_batches]
        self.video_batches = [v.to(self.device) for v in video_batches]

        self.learning_rate = config.get("learning_rate", 1e-4)
        self.num_epochs = config.get("num_epochs", 5)

        self.loss_fn = SelfSupervisedLearning().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.evaluator = EvaluatorClass(rank=0)
        self.checkpoint_dir = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/main_files/model_training"
    def start_training(self):
        print("🔁 Starting batch-based training...")
        self.model.train()
        global_step = 0
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            epoch_loss = 0.0
            for i, (audio, video) in enumerate(zip(self.audio_batches, self.video_batches)):
                self.optimizer.zero_grad()

                f_art, f_lip = self.model(audio, video)
                loss = self.loss_fn(f_art, f_lip)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                # avg_loss = running_loss / len(self.dataloader)
                # print(f"Epoch [{i + 1}], Loss: {avg_loss:.4f}")
                # ✅ Save checkpoint every 10 epochs

                save_path = os.path.join(self.checkpoint_dir, f"art_checkpoint_epoch_{i + 1}.pt")
                self.save_state(
                    # model=self.model,
                    # optimizer=self.optimizer,
                    current_epoch=epoch + 1,
                    current_loss=epoch_loss,
                    # save_path=save_path
                )
            # self.save_final_model(self.model)
            # self.start_evaluation(self.model,f_art,f_lip)
            # self.writer.close()

            # self.save_final_model(self.model)
            # self.start_evaluation(self.model,f_art,f_lip)
            # self.writer.close()

            print(f"✅ Epoch {epoch+1} Avg Loss: {epoch_loss / len(self.audio_batches):.4f}")


    def start_evaluation(self):
        print("📊 Running batch-wise evaluation...")
        self.model.eval()
        with torch.no_grad():
            for i, (audio, video) in enumerate(zip(self.audio_batches, self.video_batches)):
                print(f"  [Eval Batch {i+1}]")
                self.evaluator.evaluate(
                    model=self.model,
                    audio_inputs=audio,
                    video_inputs=video
                )

    def save_state(self,save_path = None, current_epoch=0, current_loss=0.0):
        print("💾 Saving model checkpoint...")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': current_epoch,
            'loss': current_loss
        }, save_path)

