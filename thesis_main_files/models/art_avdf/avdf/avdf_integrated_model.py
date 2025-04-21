import torch
import torch.nn as nn
from thesis_main_files.main_files.preprocessing.art_avdf.avdf.audio_preprocessor_avdf import AudioPreprocessorAVDF
from thesis_main_files.main_files.preprocessing.art_avdf.avdf.video_preprocessor_avdf import VideoPreprocessorAVDF
from thesis_main_files.main_files.feature_extraction.art_avdf.unimodal_feature_extraction import UnimodalFeatureExtractor
from thesis_main_files.models.art_avdf.art_main_module.art_model import ARTModule
from thesis_main_files.models.art_avdf.avdf.avdf_fusion_model import AudioVisualFusion
from thesis_main_files.models.art_avdf.learning_containers.final_loss_function import AVDFLoss


class AVDFModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Preprocessing components


        # Feature extraction
        self.unimodal_extractor = UnimodalFeatureExtractor()

        # Audio-visual fusion with articulatory embeddings
        self.fusion = AudioVisualFusion()

        self.art_module = ARTModule()
        # Loss computation
        self.loss_module = AVDFLoss()

    def forward(self, video_batch, audio_batch):
        # Get preprocessed inputs
        # xv = self.video_preprocessor.process_video(video_paths)
        # xa = self.audio_preprocessor.process_audio(audio_paths)

        # Extract unimodal features
        fv, fa = self.unimodal_extractor(video_batch,audio_batch)
        print(f"Visual Features (fv): {fv.shape}")  # Should be [4, dv, H, W]
        print(f"Audio Features (fa): {fa.shape}")  # Should be [4, Ta, da]
        f_art, f_dash_lip   =      self.ARTModule(video_batch, audio_batch)
        # Get articulatory features from ART module


        # Final fusion and classification
        output, f_lip_prime = self.fusion(fa, fv, f_art, f_dash_lip)

        return output

    def compute_loss(self, output, labels):
        return self.loss_module(output, labels)

# audio_path1 = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/audio_wav/train/000469.wav"
# audio_path2 = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/audio_wav/train/000470.wav"
# audio_path3 = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/audio_wav/train/000471.wav"
# audio_path4 = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/audio_wav/train/000472.wav"
#
#
# video_path1 = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/train/train/000469.mp4"
# video_path2 = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/train/train/000470.mp4"
# video_path3 = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/train/train/000471.mp4"
# video_path4 = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/train/train/000472.mp4"
# video_batch = [video_path1,video_path2,video_path3,video_path4]
# audio_batch = [audio_path1,audio_path2,audio_path3,audio_path4]
# avf = AVDFModule()
# output = AVDFModule(video_batch,audio_batch)
