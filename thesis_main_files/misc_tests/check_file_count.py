from pathlib import Path
import pandas as pd
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
def create_dataset(csv_path, num_rows=None, video_dir=None, audio_dir=None):
    df = pd.read_csv(csv_path)
    required_cols = {'video_file', 'audio_file', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV file must contain 'video_file', 'audio_file', and 'label' columns")

    if num_rows:
        df = df.iloc[:num_rows]

    video_paths = [str(Path(video_dir) / filename) for filename in df['video_file']]
    audio_paths = [str(Path(audio_dir) / filename) for filename in df['audio_file']]
    labels = df['label'].tolist()

    return audio_paths, video_paths, labels

def create_dataset_idx(csv_path, start_idx=0, num_rows=3, video_dir=None, audio_dir=None):
    df = pd.read_csv(csv_path)
    required_cols = {'video_file', 'audio_file', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV file must contain 'video_file', 'audio_file', and 'label' columns")

    # Slice the DataFrame using the provided start index and number of rows
    df = df.iloc[start_idx : start_idx + num_rows]

    video_paths = [str(Path(video_dir) / filename) for filename in df['video_file']]
    audio_paths = [str(Path(audio_dir) / filename) for filename in df['audio_file']]
    labels = df['label'].tolist()

    return audio_paths, video_paths, labels
csv_path, audio_preprocess_dir, feature_dir_audio, project_dir_video_swin, video_preprocess_dir, feature_dir_vid, audio_dir,video_dir,real_output_txt_path = convert_paths()
audio_paths, video_paths, labels = create_dataset_idx(csv_path,start_idx = 0, num_rows = 32, video_dir = video_dir, audio_dir = audio_dir)
print(audio_paths,video_paths)