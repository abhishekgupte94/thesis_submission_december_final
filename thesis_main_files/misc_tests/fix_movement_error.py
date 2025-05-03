import os
import shutil

# === Define paths ===
source_dir = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/datasets/processed/lav_df/train_fake"
destination_dir = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/datasets/processed/lav_df/train"

# === Create destination directory if it doesn't exist ===
os.makedirs(destination_dir, exist_ok=True)

# === List all files in the source directory ===
files_to_move = os.listdir(source_dir)

# === Move each file back ===
moved_files_count = 0

for file_name in files_to_move:
    source_path = os.path.join(source_dir, file_name)
    destination_path = os.path.join(destination_dir, file_name)

    if os.path.exists(destination_path):
        # Skip if already exists at destination
        print(f"Skipping (already exists): {file_name}")
        continue

    if os.path.isfile(source_path):
        shutil.move(source_path, destination_path)
        moved_files_count += 1

print(f"\nTotal files moved back: {moved_files_count}")
print(f"Moved files from {source_dir} ➔ {destination_dir}")
