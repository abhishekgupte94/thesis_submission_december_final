from torch.utils.data import Dataset

# Import your original modules for processing/extraction.
from thesis_main_files.main_files.preprocessing.art_avdf.art.video_preprocessorart_Fanet import VideoPreprocessor_FANET
from thesis_main_files.main_files.feature_extraction.art_avdf.art.feature_extractor_ART_Video import SWIN_EXECUTOR as VideoFeatureExtractor
from thesis_main_files.main_files.preprocessing.art_avdf.art.audio_preprocessorart import AudioPreprocessor
from pathlib import Path
import pandas as pd

def create_file_paths(project_dir_curr, csv_name="training_data_two.csv"):
    """
    Generates full paths for video and audio files based on filenames from a CSV file.

    Args:
        project_dir_curr (Path or str): The root project directory path.
        csv_name (str): The name of the CSV file containing filenames and labels.

    Returns:
        tuple: A tuple containing:
            - video_paths (list of Path): Full paths to video files.
            - audio_paths (list of Path): Full paths to audio files.
            - labels (list): Corresponding labels for the files.
    """
    # Ensure project_dir_curr is a Path object
    project_dir_curr = Path(project_dir_curr)

    # Define paths to the relevant directories and CSV file
    csv_path = project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / csv_name
    audio_dir = project_dir_curr / "datasets" / "processed" / "lav_df" / "audio_wav" / "train"
    video_dir = project_dir_curr / "datasets" / "processed" / "lav_df" / "train"

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Generate full paths for video and audio files
    video_paths = [video_dir / Path(filename) for filename in df['video_file']]
    audio_paths = [audio_dir / Path(filename) for filename in df['audio_file']]
    labels = df['label'].tolist()

    return video_paths, audio_paths, labels


def get_project_root(project_name=None):
    current = Path(__file__).resolve()

    # Locate the parent directory one level above 'thesis_main_files'
    for parent in current.parents:
        if parent.name == "thesis_main_files":
            base_dir = parent.parent  # One level above 'thesis_main_files'
            break
    else:
        return None  # Return None if 'thesis_main_files' is not found in the parent chain

    if project_name:
        # Search specifically for the desired project_name within the base_dir
        target_path = base_dir / project_name
        if target_path.exists() and target_path.is_dir():
            return target_path
        else:
            return None  # Return None if the specified project name is not found
    else:
        # If no project name is specified, search for known projects
        project_names = {"thesis_main_files", "Video-Swin-Transformer","melodyExtraction_JDC"}
        for parent in current.parents:
            if parent.name in project_names:
                return parent

    return None
def get_model_save_paths():
    project_dir_curr = get_project_root()
    model_save_path = str(project_dir_curr / "saved_models" / "artmodel_mac_final.pt")
    t_sne_save_path = str(project_dir_curr / "images" / "lav-df" / "evaluation_results" / "art"/"t_sne_evaluation.png")
    retrieval_save_path = str(project_dir_curr / "images" / "lav-df" / "evaluation_results" / "art"/"retrieval_evaluation.png")
    return model_save_path,t_sne_save_path,retrieval_save_path
def convert_paths():
    project_dir_curr = get_project_root()
    csv_path = str(
        project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "training_data_two.csv")
    # Audio preprocess path
    audio_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "audio_wav" / "train")
    video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" /  "train")
    audio_preprocess_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "features" / "audio" / "real")
    feature_dir_audio = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "features" / "audio" / "real")

    # Video preprocess path
    project_dir_video_swin = get_project_root("Video-Swin-Transformer")
    video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
    real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" /"lip_train_text_real.txt")

    feature_dir_vid = str(project_dir_video_swin)
    return csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin, video_preprocess_dir, feature_dir_vid, audio_dir,video_dir,real_output_txt_path
#

###############################################################################
# COMPONENT EXTRACTION CLASS
###############################################################################
class VideoComponentExtractor:
    """
    Handles extraction of raw video/audio components.
    """

    # def __init__(self):
    def extract_video_components(self, video_paths, video_save_dir, output_txt_file,batch_size,video_preprocessor): #, num_rows=None):
        try:
            # if num_rows:
            #     video_paths = video_paths[:num_rows]

            return_paths = video_preprocessor.main(video_paths)
            return return_paths.copy()
        except Exception as e:
            print(f"Error preprocessing video paths {video_paths}: {e}")
            return []


    def extract_video_component_single(self, video_path, csv_path, video_save_dir, output_txt_file):
        try:
            # if num_rows:
            #     video_paths = video_paths[:num_rows]
            video_preprocessor = VideoPreprocessor_FANET(
                batch_size=1,
                video_path=video_path,
                output_base_dir_real=video_save_dir,
                real_ouput_txt_path=output_txt_file
            )
            return_paths = video_preprocessor.main_single(video_path)
            return return_paths.copy()
        except Exception as e:
            print(f"Error preprocessing video paths {video_path}: {e}")
            return []




###############################################################################
# FEATURE EXTRACTION CLASS
###############################################################################
class VideoAudioFeatureExtractor:
    """
    Handles extraction of video/audio features from preprocessed components.
    """

    def extract_video_features(self,video_feature_extractor):#):
        try:
            # if num_rows:
                # preprocessed_paths = preprocessed_paths[:num_rows]
            # video_feature_extractor = VideoFeatureExtractor()
            features = video_feature_extractor.execute_swin()
            return features
        except Exception as e:
            print(e)
            # print(f"Error extracting features for video {preprocessed_paths}: {e}")
            return []
    #
    #
    # def extract_video_features(self):
    #     try:
    #         if num_rows:
    #             # preprocessed_paths = preprocessed_paths[:num_rows]
    #         video_feature_extractor = VideoFeatureExtractor()
    #         video_feature_single = video_feature_extractor.execute_swin()
    #         return video_feature_single
    #     except Exception as e:
    #         print(e)
    #         # print(f"Error extracting features for video {preprocessed}: {e}")
    #         return []
    def extract_audio_features(self, audio_preprocessor,audio_paths,batch_size): #,num_rows):# mel_output_dir, melody_output_dir):
        try:
            # if num_rows:
            #     audio_paths = audio_paths[:num_rows]
                # filenames = filenames[:num_rows]
            audio_features = audio_preprocessor.main_processing(audio_paths,batch_size,save_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_2_audio_embeddings/test_2_audio_embeddings.pt")
            return audio_features
        except Exception as e:
            print(f"Error extracting audio feature for {audio_paths}: {e}")
            return []
    def extract_audio_feature_single(self, audio_preprocessor,audio_path, filenames, mel_output_dir, melody_output_dir):
        try:
            # if num_rows:
            #     audio_paths = audio_paths[:num_rows]
                # filenames = filenames[:num_rows]
            audio_feature_single = audio_preprocessor.process_audio_single(audio_path)
            return audio_feature_single
        except Exception as e:
            print(f"Error extracting audio feature for {audio_path}: {e}")
            return []
###############################################################################
# PROCESSOR CLASS: COMBINING COMPONENT & FEATURE EXTRACTION
###############################################################################
class VideoAudioFeatureProcessor:
    """
    Uses separate classes for component extraction and feature extraction to create
    a dataset that contains file paths to precomputed features along with labels.
    """

    def __init__(self, video_preprocess_dir, audio_preprocess_dir, feature_dir_vid, feature_dir_audio,batch_size,video_save_dir, output_txt_file):
        self.video_preprocess_dir = video_preprocess_dir
        self.audio_preprocess_dir = audio_preprocess_dir
        self.feature_dir_vid = feature_dir_vid
        self.feature_dir_audio = feature_dir_audio
        self.audio_preprocessor = AudioPreprocessor()
        self.video_preprocessor = VideoPreprocessor_FANET(
            batch_size=batch_size,
            # video_paths=video_paths,
            output_base_dir_real=video_save_dir,
            real_output_txt_path=output_txt_file
        )
        self.video_feature_ext = VideoFeatureExtractor(video_preprocess_dir = video_preprocess_dir)
        # Instantiate the separate extractors.
        self.component_extractor = VideoComponentExtractor()
        self.feature_extractor = VideoAudioFeatureExtractor()
        self.batch_size = batch_size
        # self.video_save_dir = video_save_dir

    def create_datasubset(self, csv_path, use_preprocessed=True, video_paths=None, audio_paths=None,
                          video_save_dir=None, output_txt_file=None):
        processed_audio_features = None
        processed_video_features = None
        audio_error = False
        video_error = False

        try:
            processed_audio_features = self.feature_extractor.extract_audio_features(self.audio_preprocessor,
                                                                                     audio_paths, self.batch_size)
        except Exception as e:
            print(f"Audio Processing Error: {e}")
            audio_error = True

        try:
            preprocessed_video_paths = self.component_extractor.extract_video_components(video_paths, video_save_dir,
                                                                                         output_txt_file,
                                                                                         self.batch_size,
                                                                                         self.video_preprocessor)
        except Exception as e:
            print(f"Video Component Extraction Error: {e}")
            video_error = True

        try:
            if not video_error:
                processed_video_features = self.feature_extractor.extract_video_features(self.video_feature_ext)
        except Exception as e:
            print(f"Video Feature Extraction Error: {e}")
            video_error = True

        # Return only if there are no errors
        if not audio_error and not video_error:
            return processed_audio_features, processed_video_features
        else:
            print("Errors encountered. No features returned.")
            return None, None


###############################################################################
# DATASET CLASS FOR LOADING FEATURES
###############################################################################
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path


class VideoAudioDataset(Dataset):
    """
    Dataset class that generates video and audio paths using the create_file_paths function.
    """

    def __init__(self, project_dir_curr, csv_name="training_data_two.csv", augmentations=None):
        """
        Args:
            project_dir_curr (str or Path): Path to the root project directory.
            csv_name (str): Name of the CSV file containing the filenames and labels.
            augmentations (callable, optional): Optional transformations to be applied on a sample.
        """
        self.project_dir_curr = project_dir_curr
        self.csv_name = csv_name
        self.augmentations = augmentations

        # Generate the data (video_paths, audio_paths, labels) using create_file_paths()
        self.video_paths, self.audio_paths, self.labels = create_file_paths(project_dir_curr, csv_name)

        # Combine them into a single list of tuples for convenience
        self.data = list(zip(self.video_paths, self.audio_paths, self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve video path, audio path, and label
        video_path, audio_path, label = self.data[idx]

        # If augmentations are provided, apply them to the paths
        if self.augmentations:
            video_path = self.augmentations(video_path)
            audio_path = self.augmentations(audio_path)

        return str(video_path), str(audio_path), label

# class VideoComponentExtractorTester:
#     def __init__(self, batch_size, video_save_dir, output_txt_file):
#         self.batch_size = batch_size
#         self.video_save_dir = video_save_dir
#         self.output_txt_file = output_txt_file
#         self.video_preprocessor = VideoPreprocessor_FANET(
#             batch_size=batch_size,
#             output_base_dir_real=video_save_dir,
#             real_output_txt_path=output_txt_file
#         )
#         self.component_extractor = VideoComponentExtractor()
#
#     def test_video_component_extraction(self, video_paths):
#         try:
#             paths = self.component_extractor.extract_video_components(
#                 video_paths, self.video_save_dir, self.output_txt_file, self.batch_size, self.video_preprocessor
#             )
#             print("Video component extraction successful.")
#             return paths
#         except Exception as e:
#             print(f"Error during video component extraction: {e}")
#             return None
#
# class AudioFeatureExtractorTester:
#     def __init__(self, batch_size):
#         self.batch_size = batch_size
#         self.audio_preprocessor = AudioPreprocessor()
#         self.feature_extractor = VideoAudioFeatureExtractor()
#
#     def test_audio_feature_extraction(self, audio_paths):
#         try:
#             audio_features = self.feature_extractor.extract_audio_features(
#                 self.audio_preprocessor, audio_paths, self.batch_size
#             )
#             print("Audio feature extraction successful.")
#             return audio_features
#         except Exception as e:
#             print(f"Error during audio feature extraction: {e}")
#             return None
#
# class VideoFeatureExtractorTester:
#     def __init__(self, batch_size):
#         self.video_feature_ext = VideoFeatureExtractor(video_preprocess_dir = video_preprocess_dir)
#         self.batch_size = batch_size
#
#     def test_video_feature_extraction(self):
#         try:
#             features = self.video_feature_ext.execute_swin()
#             print("Video feature extraction successful.")
#             return features
#         except Exception as e:
#             print(f"Error during video feature extraction: {e}")
#             return None
#
#
# class VideoAudioFeatureExtractorTester:
#     def __init__(self, batch_size):
#         self.video_feature_ext = VideoFeatureExtractor(video_preprocess_dir = video_preprocess_dir)
#         self.batch_size = batch_size
#         self.audio_preprocessor = AudioPreprocessor()
#         self.feature_extractor = VideoAudioFeatureExtractor()
#     def test_both_feature_extraction(self):
#         try:
#             video_features = self.video_feature_ext.execute_swin()
#             print("Video feature extraction successful.")
#         except Exception as e:
#             print(f"Error during video feature extraction: {e}")
#
#         try:
#             audio_features = self.feature_extractor.extract_audio_features(
#                 self.audio_preprocessor, audio_paths, self.batch_size
#             )
#             print("Audio feature extraction successful.")
#         except Exception as e:
#             print(f"Error during audio feature extraction: {e}")
#             return None
#         return audio_features,video_features
#
# class VideoAudioFeatureExtractorTester:
#     def __init__(self, batch_size):
#         self.video_feature_ext = VideoFeatureExtractor(video_preprocess_dir = video_preprocess_dir)
#         self.batch_size = batch_size
#         self.audio_preprocessor = AudioPreprocessor()
#         self.feature_extractor = VideoAudioFeatureExtractor()
#
#     def test_both_feature_extraction(self, audio_paths):
#         video_features = None
#         audio_features = None
#
#         try:
#             video_features = self.video_feature_ext.execute_swin()
#             print("✅ Video feature extraction successful.")
#         except Exception as e:
#             print(f"❌ Error during video feature extraction: {e}")
#
#         try:
#             audio_features = self.feature_extractor.extract_audio_features(
#                 self.audio_preprocessor, audio_paths, self.batch_size
#             )
#             print("✅ Audio feature extraction successful.")
#         except Exception as e:
#             print(f"❌ Error during audio feature extraction: {e}")
#
#         return audio_features, video_features
#
# class VideoAudioFeatureComponentExtractorTester:
#     def __init__(self, batch_size, video_save_dir, output_txt_file):
#         self.video_feature_ext = VideoFeatureExtractor(video_preprocess_dir = video_preprocess_dir)
#         self.batch_size = batch_size
#         self.audio_preprocessor = AudioPreprocessor()
#         self.feature_extractor = VideoAudioFeatureExtractor()
#         self.video_save_dir = video_save_dir
#         self.output_txt_file = output_txt_file
#         self.video_preprocessor = VideoPreprocessor_FANET(
#             batch_size=batch_size,
#             output_base_dir_real=video_save_dir,
#             real_output_txt_path=output_txt_file
#         )
#         self.component_extractor = VideoComponentExtractor()
#
#     def test_both_feature_extraction(self, video_paths, audio_paths):
#         video_features = None
#         audio_features = None
#         extracted_paths = None
#
#         try:
#             extracted_paths = self.component_extractor.extract_video_components(
#                 video_paths, self.video_save_dir, self.output_txt_file,
#                 self.batch_size, self.video_preprocessor
#             )
#             print("✅ Video component extraction successful.")
#         except Exception as e:
#             print(f"❌ Error during video component extraction: {e}")
#
#         try:
#             video_features = self.video_feature_ext.execute_swin()
#             print("✅ Video feature extraction successful.")
#         except Exception as e:
#             print(f"❌ Error during video feature extraction: {e}")
#
#         try:
#             audio_features = self.feature_extractor.extract_audio_features(
#                 self.audio_preprocessor, audio_paths, self.batch_size
#             )
#             print("✅ Audio feature extraction successful.")
#         except Exception as e:
#             print(f"❌ Error during audio feature extraction: {e}")
#
#         return audio_features, video_features, extracted_paths
#
#
#
# if __name__ == '__main__':
#     pr = get_project_root()
#     print(pr)
#     csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin, video_preprocess_dir, feature_dir_vid, audio_dir,video_dir,real_output_txt_path = convert_paths()
#     # print(csv_path)
#
#     va = VideoAudioDataset(pr)
#
#     from torch.utils.data import DataLoader
#     batch_size = 32
#     avfet = AudioFeatureExtractorTester(batch_size)
#     # vcet = VideoComponentExtractorTester(batch_size, video_preprocess_dir,real_output_txt_path)
#     vfet = VideoFeatureExtractorTester(batch_size)
#     # vfaet = VideoAudioFeatureExtractorTester(batch_size)
#     dataloader = DataLoader(va, batch_size=32, shuffle= False, num_workers=4)
#     num = 0
#     for video_paths, audio_paths, labels in dataloader:
#         print(video_paths,audio_paths)
#         if num == 0:
#             num+=1
#             continue
#         # print(audio_paths)
#         # print(video_paths)
#         # vcet.test_video_component_extraction(video_paths)
#         # vfet.test_video_feature_extraction()
#         avfet.test_audio_feature_extraction(audio_paths)
#         # audio_features = avfet.test_audio_feature_extraction(audio_paths)
#         # print(audio_features.size())
#         # print(len(audio_features))
#         # features = vfet.test_video_feature_extraction()
#         # print(features.size())
#         # audio_features, video_features = vfaet.test_both_feature_extraction(audio_paths)
#         # video_features = vfet.test_video_feature_extraction()
#         # print(video_features.size())
#         break