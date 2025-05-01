def create_manifest_from_selected_files(selected_video_paths, output_txt_path):
    """
    Writes a text file for selected .mp4 videos with format:
    filename.mp4 0

    Args:
        selected_video_paths (list of str or Path): List of selected video file paths.
        output_txt_path (str or Path): Where to save the manifest text file.
    """
    from pathlib import Path

    output_txt_path = Path(output_txt_path)

    with output_txt_path.open("w") as f:
        for video_path in selected_video_paths:
            video_file = Path(video_path).name  # extract only the filename
            f.write(f"{video_file} 0\n")

    print(f"✅ Manifest created at: {output_txt_path} ({len(selected_video_paths)} entries)")
def preprocess_videos_for_evaluation(video_paths, output_dir, batch_size=128):
    """
    Preprocesses a list of video paths and saves lip-only videos to a common output directory.

    Args:
        video_paths (List[str]): List of raw video file paths to preprocess.
        output_dir (str): Directory where lip-only videos will be saved.
        batch_size (int): Number of frames per batch for lip extraction.
    """
    from thesis_main_files.main_files.preprocessing.art_avdf.art.video_preprocessor_SAVE_FILES_MULTI_GPU import VideoPreprocessor_FANET

    # Step 1: Initialize Preprocessor
    preproc = VideoPreprocessor_FANET(
        batch_size=batch_size,
        output_base_dir=output_dir,
        device="cuda"  # Automatically assigned per GPU by worker
    )

    # Step 2: Run preprocessing in parallel across GPUs
    preproc.parallel_main(video_paths)

    print(f"✅ All videos preprocessed and saved to: {output_dir}")
