import os
import shutil
import pandas as pd

# === Define paths ===
training_data_path = "/thesis_main_files/data/processed_files/csv/lav_df/training_data/training_data_one.csv"
source_dir = "/thesis_main_files/data/processed_files/lav_df/train"
destination_dir = "/thesis_main_files/data/processed_files/lav_df/train_fake"

# === Create destination directory if it doesn't exist ===
os.makedirs(destination_dir, exist_ok=True)

# === Load training_data_one.csv safely (preserving leading zeros) ===
training_data = pd.read_csv(training_data_path, dtype=str, keep_default_na=False)

# === Ensure 'label' and 'filename' columns exist ===
if 'label' not in training_data.columns or 'filename' not in training_data.columns:
    raise ValueError("The dataset must contain 'label' and 'filename' columns.")

# === Convert label to integer for safe filtering ===
training_data['label'] = training_data['label'].astype(int)

# === Filter fake filenames ===
fake_filenames = training_data[training_data['label'] == 1]['filename'].tolist()

# === Calculate total size of fake video to move ===
total_size_to_move = 0
for filename in fake_filenames:
    video_filename = f"{filename}.mp4"
    source_path = os.path.join(source_dir, video_filename)
    if os.path.exists(source_path):
        total_size_to_move += os.path.getsize(source_path)

# === Check available disk space ===
destination_disk = shutil.disk_usage(destination_dir)
available_space = destination_disk.free

if total_size_to_move > available_space:
    raise RuntimeError(f"Not enough disk space! Needed: {total_size_to_move/1e9:.2f} GB, Available: {available_space/1e9:.2f} GB.")

print(f"Enough space available. Proceeding to move {len(fake_filenames)} fake video files...")

# === Move matching files ===
moved_files_count = 0

for filename in fake_filenames:
    video_filename = f"{filename}.mp4"
    source_path = os.path.join(source_dir, video_filename)
    destination_path = os.path.join(destination_dir, video_filename)

    if os.path.exists(destination_path):
        # Skip if already exists at destination
        print(f"Skipping (already exists): {video_filename}")
        continue

    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        moved_files_count += 1
    else:
        print(f"Warning: {video_filename} not found in {source_dir}")

print(f"\nTotal fake videos moved: {moved_files_count}")
print(f"Moved files from {source_dir} ➔ {destination_dir}")
