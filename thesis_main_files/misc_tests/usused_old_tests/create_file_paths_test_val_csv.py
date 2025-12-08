import os
import json
import pandas as pd

# === Paths ===
metadata_path = "/thesis_main_files/data/metadata/lav-df/metadata.json"

source_test_dir = "/Users/abhishekgupte_macbookpro/Downloads/Datasets/LAV-DF/test"
source_dev_dir = "/Users/abhishekgupte_macbookpro/Downloads/Datasets/LAV-DF/dev"

destination_test_dir = "/thesis_main_files/data/processed_files/lav_df/test"
destination_dev_dir = "/thesis_main_files/data/processed_files/lav_df/dev"

# === Output CSV path ===
output_csv_path = "/thesis_main_files/data/processed_files/lav_df/filename_to_paths.csv"

# === Load metadata.json ===
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# === List to collect data ===
data_rows = []

# === Loop through metadata entries ===
for entry in metadata:
    file_path = entry['file']  # Example: "test/000001.mp4" or "dev/000015.mp4"

    if file_path.startswith('test/'):
        filename = file_path.replace('test/', '')  # "000001.mp4"
        source_path = os.path.join(source_test_dir, filename)
        destination_path = os.path.join(destination_test_dir, filename)
        split = "TEST"
    elif file_path.startswith('dev/'):
        filename = file_path.replace('dev/', '')
        source_path = os.path.join(source_dev_dir, filename)
        destination_path = os.path.join(destination_dev_dir, filename)
        split = "DEV"
    else:
        continue  # Ignore unrelated entries

    # Append to list
    data_rows.append({
        'filename': filename,
        'destination_filepath': destination_path,
        'source_filepath': source_path,
        'split': split
    })

# === Create dataframe ===
df = pd.DataFrame(data_rows)

# === Save to CSV ===
df.to_csv(output_csv_path, index=False)

# === Done ===
print(f"Saved {len(df)} entries to {output_csv_path}")
