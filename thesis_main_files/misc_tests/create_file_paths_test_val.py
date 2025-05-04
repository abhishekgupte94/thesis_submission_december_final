import os
import shutil
import pandas as pd

# === Paths ===
csv_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/datasets/processed/lav_df/filename_to_paths.csv"
move_failures_log = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/datasets/processed/lav_df/move_failures.txt"

# === Load CSV ===
df = pd.read_csv(csv_path)

# === Track failures ===
failures = []

# === Move operation ===
for idx, row in df.iterrows():
    source_path = row['source_filepath']
    destination_path = row['destination_filepath']

    # Create destination directory if it does not exist
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Check if source exists
    if not os.path.exists(source_path):
        print(f"Warning: Source file does not exist: {source_path}")
        failures.append(f"Missing Source: {source_path}")
        continue

    try:
        shutil.move(source_path, destination_path)

        # Verify the move: destination must exist after moving
        if not os.path.exists(destination_path):
            print(f"Warning: Failed to move file: {source_path}")
            failures.append(f"Move Failed: {source_path} ➔ {destination_path}")

    except Exception as e:
        print(f"Exception occurred while moving {source_path}: {str(e)}")
        failures.append(f"Exception: {source_path} ➔ {destination_path} :: {str(e)}")

# === Save failures if any ===
if failures:
    with open(move_failures_log, 'w') as f:
        for fail in failures:
            f.write(fail + "\n")
    print(f"\nSome files failed to move. Logged {len(failures)} issues into {move_failures_log}")
else:
    print("\nAll files moved successfully!")
