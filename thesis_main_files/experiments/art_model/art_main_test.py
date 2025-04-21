from thesis_main_files.main_files.model_training.train_art_wrapper_test import TestTrainingPipelineWrapper
import torch
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
audio_path_one = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_1_audio_embeddings/test_1_audio_embeddings.pt"
audio_path_two = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_2_audio_embeddings/test_2_audio_embeddings.pt"
video_path_one = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_1_video_embeddings/batch_features_lips.pt"
video_path_two = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_2_video_embeddings/batch_features_lips.pt"

audio_paths = [audio_path_two, audio_path_two]
video_paths = [video_path_one, video_path_two]
audio_batches = [torch.load(p) for p in audio_paths]
video_batches = [torch.load(p) for p in video_paths]

# Instantiate model
model = ARTModule()

# Config
config = {
    "learning_rate": 1e-4,
    "num_epochs": 5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Run test-time training & evaluation
wrapper = TestTrainingPipelineWrapper(model, audio_batches, video_batches, config=config)
wrapper.start_training()
wrapper.start_evaluation()
