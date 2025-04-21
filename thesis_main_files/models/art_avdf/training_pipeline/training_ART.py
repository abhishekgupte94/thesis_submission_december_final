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
#
# # Step 3: Initialize the Processor with needed dirs and batch_size
# batch_size = 128
#
# processor = VideoAudioFeatureProcessor(
#     video_preprocess_dir=video_preprocess_dir,
#     audio_preprocess_dir=audio_preprocess_dir,
#     feature_dir_vid=feature_dir_vid,
#     feature_dir_audio=feature_dir_audio,
#     batch_size=batch_size,
#     video_save_dir=video_preprocess_dir,
#     output_txt_file=real_output_txt_path
# )
#
# # Step 5: Create a dataset class for raw file loading (not yet using features, just paths + labels)
# dataset = VideoAudioDataset(
#     project_dir_curr=get_project_root(),
#     csv_name="training_data_two.csv"
# )

class TrainingPipeline:
    def __init__(self, dataset, batch_size, learning_rate, num_epochs, device, feature_processor):
        """
        Args:
            model (nn.Module): The model to be trained (ModelA).
            dataset (VideoAudioDataset): The Dataset instance.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            num_epochs (int): Number of training epochs.
            device (torch.device): Device to train on ('cuda' or 'cpu').
            feature_processor (VideoAudioFeatureProcessor): The feature extraction class instance.
        """
        self.model = ARTModule()
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.device = device
        self.loss_fn = SelfSupervisedLearning()
        self.feature_processor = feature_processor  # The feature extraction processor
    def train(self):
        self.model.train()

        log_dir = f"runs/ssl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        writer = SummaryWriter(log_dir=log_dir)
        global_step = 0

        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for video_paths, audio_paths, labels in self.dataloader:

                # Process features using the feature processor
                processed_audio_features, processed_video_features = self.feature_processor.create_datasubset(
                    csv_path=None,  # Not needed, we already have the paths
                    use_preprocessed=False,
                    video_paths=video_paths,
                    audio_paths=audio_paths
                )

                if processed_audio_features is None or processed_video_features is None:
                    print("Skipping batch due to feature extraction failure.")
                    continue  # Skip this batch if feature extraction fails

                # Convert features to tensors and move them to the device
                audio_features = torch.stack(processed_audio_features).to(self.device)
                video_features = torch.stack(processed_video_features).to(self.device)
                # labels = torch.tensor(labels).to(self.device)

                f_art, f_lip = self.model(audio_features, video_features)
                loss = self.loss_fn(f_art, f_lip)
                # Forward pass
                self.optimizer.zero_grad()
                # loss, fl_prime, fa_refined,pos_sim, neg_sim, temperature,similarity_matrix = self.model.training_step(video_features, audio_features)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                # 📊 TensorBoard Logging
                # writer.add_scalar("Loss/train", loss.item(), global_step)
                # writer.add_scalar("Similarity Matrix", similarity_matrix.item(), global_step)
                # writer.add_scalar("Similarity/Negative", neg_sim.item(), global_step)
                # writer.add_scalar("Temperature", temperature.item(), global_step)
                # Backward pass and optimization
                global_step+=1
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(self.dataloader):.4f}")

        print("Training completed.")
        writer.close()  # Don't forget to close!
