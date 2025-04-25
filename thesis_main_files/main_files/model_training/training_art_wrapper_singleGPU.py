import torch
from thesis_main_files.models.art_avdf.training_pipeline.training_ART_singleGPU_final_annotated import TrainingPipeline
from thesis_main_files.models.data_loaders.data_loader_ART import (
    VideoAudioFeatureProcessor,
    VideoAudioDataset,
    create_file_paths,
    get_project_root,
    convert_paths
)
from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass
from pathlib import Path
class TrainingPipelineWrapper:
    def __init__(self, config=None):
        """
        Initializes and prepares the training pipeline for single-GPU training.

        Args:
            config (dict, optional): Configuration dictionary. Keys can include:
                - batch_size (int)
                - learning_rate (float)
                - num_epochs (int)
                - csv_name (str)
        """
        if config is None:
            config = {}

        # Load paths from utilities
        # (csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin,
        #  video_preprocess_dir, feature_dir_vid, audio_dir, video_dir,
        #  real_output_txt_path) = convert_paths()
        (csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path) = convert_paths()
        # Create dataset file paths
        from pathlib import Path
        csv_name = config.get("csv_name", "training_data_two.csv")
        video_paths, labels = create_file_paths(get_project_root(), csv_name)
        self.evaluator = EvaluatorClass(device = config.get("device"))
        # Prepare the feature processor
        batch_size = config.get("batch_size", 128)
        processor = VideoAudioFeatureProcessor(
            video_preprocess_dir=video_preprocess_dir,
            # audio_preprocess_dir=audio_preprocess_dir,
            feature_dir_vid=feature_dir_vid,
            # feature_dir_audio=feature_dir_audio,
            batch_size=batch_size,
            video_save_dir=video_preprocess_dir,
            output_txt_file=real_output_txt_path
        )

        # Prepare the dataset
        dataset = VideoAudioDataset(
            project_dir_curr= get_project_root(),
            csv_name=csv_name
        )

        # Training params
        learning_rate = config.get("learning_rate", 1e-4)
        num_epochs = config.get("num_epochs", 300)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Evaluator (optional for post-training)
        # self.evaluator = EvaluatorClass()

        # Initialize single-GPU training pipeline
        self.pipeline = TrainingPipeline(
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=device,
            feature_processor=processor,
            output_txt_path=real_output_txt_path
        )
        # self.device = device
        self.evaluator = EvaluatorClass(device = device)
    def start_training(self,checkpoint_dir):
        print("Starting training on single GPU...")
        self.pipeline.train(checkpoint_dir)
        print("✅ Training complete.")



    def save_final_model(self, model, save_path="final_trained_model.pt"):
        """
        Save the final trained model (for deployment or inference).
        """
        torch.save(model, save_path)
        print(f"✅ Final trained model saved to: {save_path}")

    def start_evaluation(self, model, audio_inputs, video_inputs):
        """
        Run evaluation using the evaluator.
        """
        self.evaluator.evaluate(model, audio_inputs, video_inputs)
        print("✅ Evaluation complete.")
class TrainingPipelineWrapper_DFDC:
    def __init__(self, config=None):
        """
        Initializes and prepares the training pipeline for single-GPU training.

        Args:
            config (dict, optional): Configuration dictionary. Keys can include:
                - batch_size (int)
                - learning_rate (float)
                - num_epochs (int)
                - csv_name (str)
        """
        if config is None:
            config = {}

        # Load paths from utilities
        # (csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin,
        #  video_preprocess_dir, feature_dir_vid, audio_dir, video_dir,
        #  real_output_txt_path) = convert_paths()
        (csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path) = convert_paths()
        # Create dataset file paths
        csv_name = config.get("csv_name_dfdc", "training_data_two.csv")
        video_paths, labels = create_file_paths(Path(get_project_root(),
                                                 csv_name))

        # Prepare the feature processor
        batch_size = config.get("batch_size", 128)
        processor = VideoAudioFeatureProcessor(
            video_preprocess_dir=video_preprocess_dir,
            # audio_preprocess_dir=audio_preprocess_dir,
            feature_dir_vid=feature_dir_vid,
            # feature_dir_audio=feature_dir_audio,
            batch_size=batch_size,
            video_save_dir=video_preprocess_dir,
            output_txt_file=real_output_txt_path
        )

        # Prepare the dataset
        dataset = VideoAudioDataset(
            project_dir_curr=get_project_root(),
            csv_name=csv_name
        )

        # Training params
        learning_rate = config.get("learning_rate", 1e-4)
        num_epochs = config.get("num_epochs", 300)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Evaluator (optional for post-training)

        # Initialize single-GPU training pipeline
        self.pipeline = TrainingPipeline(
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=device,
            feature_processor=processor
        )
        self.device = device
        self.evaluator = EvaluatorClass(device = device)
    def start_training(self,checkpoint_dir):
        print("Starting training on single GPU...")
        self.pipeline.train(checkpoint_dir)
        print("✅ Training complete.")
    def start_evaluation(self, model, audio_inputs, video_inputs):
        """
        Run evaluation using the evaluator.
        """
        self.evaluator.evaluate(model, audio_inputs, video_inputs)
        print("✅ Evaluation complete.")

    def save_final_model(self, model, save_path="final_trained_model.pt"):
        """
        Save the final trained model (for deployment or inference).
        """
        torch.save(model, save_path)
        print(f"✅ Final trained model saved to: {save_path}")

    # def start_evaluation(self, model, audio_inputs, video_inputs):
    #     """
    #     Run evaluation using the evaluator.
    #     """
    #     self.evaluator.evaluate(model, audio_inputs, video_inputs)
    #     print("✅ Evaluation complete.")
    #
    # def save_final_model(self, model, save_path="final_trained_model.pt"):
    #     """
    #     Save the final trained model (for deployment or inference).
    #     """
    #     torch.save(model, save_path)
    #     print(f"✅ Final trained model saved to: {save_path}")