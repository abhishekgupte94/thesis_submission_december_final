import torch
import torch.nn as nn
# from main_files.preprocessing.art_avdf.art.audio_preprocessorart import AudioPreprocessor
# from main_files.preprocessing.art_avdf.art.video_preprocessorart_Fanet_gpu import VideoPreprocessor_FANET
from thesis_main_files.models.art_avdf.encoders.audio_articulatory_encoder import AudioArticulatoryEncoder
from thesis_main_files.models.art_avdf.encoders.video_articulatory_encoder import VideoArticulatoryEncoder
# from main_files.preprocessing.art_avdf.avdf.audio_preprocessor_avdf import AudioPreprocessorAVDF
# from main_files.preprocessing.art_avdf.avdf.video_preprocessor_avdf import VideoPreprocessorAVDF
from thesis_main_files.models.art_avdf.learning_containers.self_supervised_learning import  SelfSupervisedLearning
# from main_files
audio_path = "/datasets/processed/lav_df/audio_wav/train/000471.wav"
video_path = "/Users/abhishekgupte_macbookpro/Downloads/Datasets/LAV-DF/train_filenames/000471.mp4"

class ARTModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Preprocessing components
        # self.audio_preprocessor = AudioPreprocessorAVDF()
        # Feature extraction and encoding components
        # self.feature_extractor = FeatureExtractorART()
        self.articulatory_encoder = AudioArticulatoryEncoder()
        self.video_encoder = VideoArticulatoryEncoder()
        # Self-supervised learning
        self.ssl = SelfSupervisedLearning()

    def forward(self,audio_batch,video_batch):

        f_art = self.articulatory_encoder(audio_batch)
        f_prime_lip = self.video_encoder(video_batch)
        # return art_features_batch,lip_features_batch
        return f_art,f_prime_lip