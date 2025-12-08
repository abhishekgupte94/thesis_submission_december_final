import pandas as pd
import os

# === Paths ===
base_dir = "/thesis_main_files/data/processed_files/csv/lav_df/training_data"

files = [
    "training_data_test.csv",
    "training_data_dev.csv",
    "training_data_test_and_val.csv"
]

# === Loop through each file ===
for filename in files:
    file_path = os.path.join(base_dir, filename)

    # === Load file ===
    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        print(f"Loaded {filename} with {len(df)} rows.")
    except Exception as e:
        raise RuntimeError(f"Failed to load {filename}: {str(e)}")

    # === Add or Update 'label' column ===
    try:
        if 'n_fakes' not in df.columns:
            raise ValueError(f"'n_fakes' column missing in {filename}, cannot create 'label'.")

        # Ensure n_fakes is numeric
        df['n_fakes'] = df['n_fakes'].astype(int)

        # Create or overwrite label
        df['label'] = df['n_fakes'].apply(lambda x: 0 if x == 0 else 1)

        # Save back
        df.to_csv(file_path, index=False, quoting=1)  # quoting=1 -> QUOTE_NONNUMERIC
        print(f"Updated and saved {filename} with 'label' column.")
    except Exception as e:
        raise RuntimeError(f"Failed to update {filename}: {str(e)}")
