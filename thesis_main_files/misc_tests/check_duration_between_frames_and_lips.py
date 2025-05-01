import os
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import re


def get_video_duration(file_path):
    try:
        if os.path.exists(file_path):
            video = VideoFileClip(file_path)
            duration = video.duration  # Duration in seconds
            video.close()
            return duration
        else:
            print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def get_video_files(directory):
    video_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                video_files[file] = full_path  # Store full path for duration calculation
    return video_files


def calculate_durations(video_files):
    durations = {}
    stripped_durations = {}  # New dictionary for storing stripped-down filenames

    for filename, full_path in video_files.items():
        duration = get_video_duration(full_path)
        if duration is not None:
            durations[filename] = duration

            # Extract stripped-down filename (Sa.mp4)
            stripped_name = re.match(r"([a-zA-Z0-9]+)_.*\.mp4", filename)
            if stripped_name:
                stripped_filename = f"{stripped_name.group(1)}.mp4"
                stripped_durations[stripped_filename] = duration
        else:
            print(f"Duration calculation failed for: {filename}")

    return durations, stripped_durations


# Directories
dir1 = '/Users/abhishekgupte_macbookpro/PycharmProjects/Video-Swin-Transformer/data/train'

# Get video files from directory 1
files_dir1 = get_video_files(dir1)
durations1, stripped_durations1 = calculate_durations(files_dir1)

# Load the CSV file
csv_file = '/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/csv_files/lav_df/inference_data/training_data_two.csv'  # Update this with your actual CSV file path
df = pd.read_csv(csv_file)

# Filter CSV to only include rows where 'fake_periods' is not empty
filtered_df = df[df['fake_periods'].notna() & (df['fake_periods'].astype(str).str.strip() != '')]

# Create a new dictionary to store the matched data
matched_data = []

# Compare and match the stripped filenames
for stripped_filename, dir1_duration in stripped_durations1.items():
    # Find the row in the filtered CSV where the 'file' matches the stripped filename
    matching_row = filtered_df[filtered_df['file'] == stripped_filename]

    if not matching_row.empty:
        csv_duration = matching_row.iloc[0]['duration']
        fake_periods = matching_row.iloc[0]['fake_periods']

        matched_data.append([stripped_filename, csv_duration, dir1_duration, fake_periods])

# Create a DataFrame for the filtered comparison
comparison_df = pd.DataFrame(matched_data, columns=['Filename', 'CSV Duration', 'Directory 1 Duration', 'Fake Periods'])

# Display the filtered comparison DataFrame
print(comparison_df.to_string(index=False))
