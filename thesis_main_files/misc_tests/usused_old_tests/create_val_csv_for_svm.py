import pandas as pd
import shutil
import os

# Define file paths
input_csv_path = "/thesis_main_files/data/processed_files/csv/lav_df/training_data/training_data_test_and_val.csv"
output_csv_path = "/thesis_main_files/data/processed_files/csv/lav_df/training_data/val_data_for_svm.csv"
source_base_path = "/thesis_main_files/data/processed_files/lav_df/dev"
destination_base_path = "/thesis_main_files/data/processed_files/lav_df/checks/data_to_preprocess_for_svm_val"
missing_files_log_path = os.path.join(destination_base_path, "missing_files_log.txt")

# Load the dataset
df = pd.read_csv(input_csv_path)

# Filter rows where 'file' starts with 'dev/'
val_df = df[df['file'].str.startswith('dev/')]

# Save the filtered dataframe to a new CSV file
val_df.to_csv(output_csv_path, index=False)

print(f"Saved {len(val_df)} validation rows to {output_csv_path}")

# Ensure destination directory exists
os.makedirs(destination_base_path, exist_ok=True)

# Track missing files
missing_files = []

# Move matched files
for file_path in val_df['file']:
    filename = os.path.basename(file_path)
    source_file = os.path.join(source_base_path, filename)
    destination_file = os.path.join(destination_base_path, filename)

    if os.path.exists(source_file):
        shutil.copy2(source_file, destination_file)
    else:
        missing_files.append(filename)

# Save missing files log if there are any
if missing_files:
    with open(missing_files_log_path, 'w') as f:
        for missing_file in missing_files:
            f.write(f"{missing_file}\n")
    print(f"Warning: {len(missing_files)} files were missing. Log saved to {missing_files_log_path}")
else:
    print("All files successfully copied.")
