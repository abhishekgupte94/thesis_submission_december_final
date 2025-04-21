# from torch.utils.data import DataLoader
# from data_loader_ART import create_file_paths,convert_paths,get_project_root,VideoComponentExtractor,VideoAudioFeatureExtractor,VideoAudioFeatureProcessor,VideoAudioDataset
# # Step 1: Convert relative paths
# csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin, video_preprocess_dir, feature_dir_vid, audio_dir, video_dir, real_output_txt_path = convert_paths()
#
# # Step 2: Create file paths for video/audio + labels
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
# # Step 4: Extract features (both audio and video)
# processed_audio_features, processed_video_features = processor.create_datasubset(
#     csv_path=csv_path,
#     use_preprocessed=False,
#     video_paths=video_paths,
#     audio_paths=audio_paths,
#     video_save_dir=video_preprocess_dir,
#     output_txt_file=real_output_txt_path
# )
#
# # Step 5: Create a dataset class for raw file loading (not yet using features, just paths + labels)
# dataset = VideoAudioDataset(
#     project_dir_curr=get_project_root(),
#     csv_name="training_data_two.csv"
# )
#
# # Step 6: Wrap in DataLoader for batching
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
#
