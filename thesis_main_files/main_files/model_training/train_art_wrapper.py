# from thesis_main_files.models.art_avdf.training_pipeline.training_ART import TrainingPipeline
# from thesis_main_files.models.data_loaders.data_loader_ART import VideoAudioFeatureProcessor,VideoAudioDataset
# from thesis_main_files.config import CONFIG
# from thesis_main_files.main_files.evaluation.art.evaluator import EvaluatorClass
# from thesis_main_files.models.data_loaders.data_loader_ART import VideoAudioFeatureProcessor,VideoAudioDataset,create_file_paths,get_project_root,convert_paths, get_model_save_paths
# import torch
#
# class TrainingPipelineWrapper:
#     def __init__(self, model, config=None):
#         """
#         Initializes and prepares the training pipeline from scratch, including data loading and feature processors.
#
#         Args:
#             model (nn.Module): The model to train.
#             config (dict, optional): Configuration dictionary. Keys can include:
#                 - batch_size (int)
#                 - learning_rate (float)
#                 - num_epochs (int)
#                 - device (torch.device)
#                 - csv_name (str)
#         """
#         if config is None:
#             config = {}
#
#         # Load paths from utilities
#         (csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin,
#          video_preprocess_dir, feature_dir_vid, audio_dir, video_dir,
#          real_output_txt_path) = convert_paths()
#
#         # Create dataset file paths
#         csv_name = config.get("csv_name", "training_data_two.csv")
#         video_paths, audio_paths, labels = create_file_paths(get_project_root(), csv_name)
#
#         # Prepare the feature processor
#         batch_size = config.get("batch_size", 128)
#         processor = VideoAudioFeatureProcessor(
#             video_preprocess_dir=video_preprocess_dir,
#             audio_preprocess_dir=audio_preprocess_dir,
#             feature_dir_vid=feature_dir_vid,
#             feature_dir_audio=feature_dir_audio,
#             batch_size=batch_size,
#             video_save_dir=video_preprocess_dir,
#             output_txt_file=real_output_txt_path
#         )
#
#         # Prepare the dataset
#         dataset = VideoAudioDataset(
#             project_dir_curr=get_project_root(),
#             csv_name=csv_name
#         )
#
#         # Final device and training params
#         learning_rate = config.get("learning_rate", 1e-4)
#         num_epochs = config.get("num_epochs", 10)
#         device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         self.evaluator = EvaluatorClass()
#         # Instantiate the actual training pipeline
#         self.pipeline = TrainingPipeline(
#             model=model,
#             dataset=dataset,
#             batch_size=batch_size,
#             learning_rate=learning_rate,
#             num_epochs=num_epochs,
#             device=device,
#             feature_processor=processor
#         )
#
#     def start_training(self):
#         """
#         Kicks off the training loop.
#         """
#         print("Starting training...")
#         self.pipeline.train()
#         print("Training complete.")
#
#     def save_state(self, model, optimizer, current_epoch, current_loss, save_path):
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'epoch': current_epoch,
#             'loss': current_loss
#         }, "model_checkpoint.pt")
#         print(f"✅ Model checkpoint saved to {save_path}")
#
#     def save_final_model(self, model, save_path):
#         torch.save(model, save_path)
#         print(f"✅ Final trained model saved to: {save_path}")
#     def start_evaluation(self,model,audio_inputs,video_inputs,t_sne_save_path,retrieval_save_path):
#         self.evaluator.evaluate(model,audio_inputs,video_inputs,t_sne_save_path,retrieval_save_path)
#         print("Evaluation complete.")
#
