import os
import json
import pandas as pd

# === Paths ===
metadata_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/datasets/metadata/lav-df/metadata.json"
output_dir = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/datasets/processed/csv_files/lav_df/training_data"
output_csv_path = os.path.join(output_dir, "training_data_fakes_train_only.csv")

# === Create output dir if not exists ===
os.makedirs(output_dir, exist_ok=True)

# === Load metadata.json ===
try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded metadata with {len(metadata)} entries.")
except Exception as e:
    raise RuntimeError(f"Failed to load metadata.json: {str(e)}")

# === Filter: train/ entries with n_fakes >= 1 ===
fake_train_entries = []

for entry in metadata:
    file_field = entry.get('file', '')

    if file_field.startswith('train/'):
        # Extract filename
        filename = file_field.replace('train/', '')

        # Get n_fakes safely
        n_fakes = int(entry.get('n_fakes', 0))

        if n_fakes >= 1:
            # Prepare entry
            new_entry = entry.copy()
            base_filename = os.path.splitext(filename)[0]  # remove .mp4
            new_entry['video_file'] = f"{base_filename}.mp4"
            new_entry['audio_file'] = f"{base_filename}.wav"
            new_entry['label'] = 1  # Explicit since it's fake
            fake_train_entries.append(new_entry)

print(f"Total fake train entries found: {len(fake_train_entries)}")

# === Save to CSV ===
if fake_train_entries:
    df_fakes = pd.DataFrame(fake_train_entries)
    df_fakes.to_csv(output_csv_path, index=False, quoting=1)  # quoting=1 -> QUOTE_NONNUMERIC
    print(f"Saved fake train data to {output_csv_path}")
else:
    print("No fake train entries found to save.")


# === Checker function ===
def check_fake_train_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        df['n_fakes'] = df['n_fakes'].astype(int)
        df['label'] = df['label'].astype(int)

        total_rows = len(df)
        valid_fakes = (df['n_fakes'] >= 1).sum()
        correct_labels = (df['label'] == 1).sum()

        print("\nChecker Summary:")
        print(f"Total rows: {total_rows}")
        print(f"Rows with n_fakes >= 1: {valid_fakes}")
        print(f"Rows with label == 1: {correct_labels}")

        if (total_rows == valid_fakes) and (total_rows == correct_labels):
            print("✅ Checker Passed: All rows are valid fake training entries.")
        else:
            print("❌ Checker Failed: There are inconsistencies.")

    except Exception as e:
        raise RuntimeError(f"Failed to run checker: {str(e)}")


# === Run Checker ===
check_fake_train_csv(output_csv_path)
