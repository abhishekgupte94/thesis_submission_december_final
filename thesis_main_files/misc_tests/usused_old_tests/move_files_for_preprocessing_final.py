import os
import pandas as pd
import shutil
from tqdm import tqdm

# === Paths ===
base_dir = "/thesis_main_files/data/processed_files/lav_df"
csv_dir = os.path.join(base_dir, "checks")
os.makedirs(csv_dir, exist_ok=True)

# Source folders
train_fake_dir = os.path.join(base_dir, "train_fake")
dev_dir = os.path.join(base_dir, "dev")
test_dir = os.path.join(base_dir, "test")

# Paths
training_svm_csv_path = "/thesis_main_files/data/processed_files/csv/lav_df/training_data/final_training_data_svm/training_data_svm_final.csv"

filtered_csv_path = os.path.join(csv_dir, "filtered_training_data_svm_existing_files.csv")
destination_dir = os.path.join(csv_dir, "data_to_preprocess_for_svm")
os.makedirs(destination_dir, exist_ok=True)

# Log files
missing_files_log_path = os.path.join(csv_dir, "missing_files.txt")
failed_copies_log_path = os.path.join(csv_dir, "failed_copies.txt")

# === Step 1: Filter only matched files ===

# Load original SVM CSV
df_svm = pd.read_csv(training_svm_csv_path, dtype=str, keep_default_na=False)

filtered_rows = []
missing_files = []

print(f"\n🔍 Checking physical existence of files...")

for idx, row in tqdm(df_svm.iterrows(), total=len(df_svm), desc="Checking Files", unit="file"):
    file_entry = row['file']

    if '/' in file_entry:
        prefix, pure_filename = file_entry.split('/', 1)
    else:
        prefix = None
        pure_filename = file_entry

    if prefix == "test":
        source_dir = test_dir
        expected_dir_name = "test"
    elif prefix == "dev":
        source_dir = dev_dir
        expected_dir_name = "dev"
    elif prefix == "train":
        source_dir = train_fake_dir
        expected_dir_name = "train_fake"
    else:
        # Unknown prefix or missing prefix
        missing_files.append(f"UNKNOWN: {file_entry}")
        continue

    full_source_path = os.path.join(source_dir, pure_filename)

    if os.path.isfile(full_source_path):
        row['source_path'] = full_source_path
        filtered_rows.append(row)
    else:
        missing_files.append(f"{expected_dir_name}/{pure_filename}")

# Create filtered DataFrame
df_filtered = pd.DataFrame(filtered_rows)

# Save filtered CSV
df_filtered.to_csv(filtered_csv_path, index=False)
print(f"\n✅ Step 1 complete: Filtered dataset saved with {len(df_filtered)} rows at {filtered_csv_path}")

# Save missing file entries with expected directory
if missing_files:
    with open(missing_files_log_path, "w") as f:
        for missing in missing_files:
            f.write(missing + "\n")
    print(f"⚠️ {len(missing_files)} missing files logged to {missing_files_log_path}")

# === Step 2: Copy matched files ===

missing_copy = []

print(f"\n🔄 Copying matched files to {destination_dir} ...")

# COPY ONLY FROM df_filtered
for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Copying Matched Files", unit="file"):
    src = row['source_path']
    dst = os.path.join(destination_dir, row['file'].split('/')[-1])

    try:
        shutil.copy2(src, dst)  # Overwrite if already exists
        if not os.path.exists(dst):
            print(f"⚠️ Copy verification failed for {dst}")
            missing_copy.append(row['file'])
    except Exception as e:
        print(f"⚠️ Failed to copy {src} -> {dst}: {str(e)}")
        missing_copy.append(row['file'])

print(f"\n✅ Step 2 complete: Copied {len(df_filtered) - len(missing_copy)} files to {destination_dir}")

# Save failed copies if any
if missing_copy:
    with open(failed_copies_log_path, "w") as f:
        for missing in missing_copy:
            f.write(missing + "\n")
    print(f"⚠️ {len(missing_copy)} failed copies logged to {failed_copies_log_path}")
