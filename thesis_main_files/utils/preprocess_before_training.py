def preprocess_videos_before_training(csv_path, csv_column, output_dir, batch_size=128):
    """
    Reads video paths from a CSV file and preprocesses them into a common output directory.

    Args:
        csv_path (str): Path to the CSV file.
        csv_column (str): Column in CSV containing video paths.
        output_dir (str): Directory where lip-only videos will be saved.
        batch_size (int): Number of frames per batch for lip extraction.
    """
    import pandas as pd
    from thesis_main_files.main_files.preprocessing.art_avdf.art.video_preprocessor_SAVE_FILES_MULTI_GPU import VideoPreprocessor_FANET

    # Step 1: Read CSV
    df = pd.read_csv(csv_path)
    if csv_column not in df.columns:
        raise ValueError(f"Column '{csv_column}' not found in {csv_path}")

    video_paths = df[csv_column].tolist()

    # Step 2: Initialize Preprocessor
    preproc = VideoPreprocessor_FANET(
        batch_size=batch_size,
        output_base_dir=output_dir,
        device="cuda"  # auto-handled per rank
    )

    # Step 3: Preprocess all videos
    preproc.parallel_main(video_paths)

    print(f"✅ All videos preprocessed and saved to: {output_dir}")

def preprocess_videos_before_evaluation(csv_path, csv_column, output_dir, batch_size=128):
    """
    Reads video paths from a CSV file and preprocesses them into a common output directory.

    Args:
        csv_path (str): Path to the CSV file.
        csv_column (str): Column in CSV containing video paths.
        output_dir (str): Directory where lip-only videos will be saved.
        batch_size (int): Number of frames per batch for lip extraction.
    """
    import pandas as pd
    from thesis_main_files.main_files.preprocessing.art_avdf.art.video_preprocessor_SAVE_FILES_MULTI_GPU import VideoPreprocessor_FANET

    # Step 1: Read CSV
    df = pd.read_csv(csv_path)
    if csv_column not in df.columns:
        raise ValueError(f"Column '{csv_column}' not found in {csv_path}")

    video_paths = df[csv_column].tolist()

    # Step 2: Initialize Preprocessor
    preproc = VideoPreprocessor_FANET(
        batch_size=batch_size,
        output_base_dir=output_dir,
        device="cuda"  # auto-handled per rank
    )

    # Step 3: Preprocess all videos
    preproc.parallel_main(video_paths)

    print(f"✅ All videos preprocessed and saved to: {output_dir}")
