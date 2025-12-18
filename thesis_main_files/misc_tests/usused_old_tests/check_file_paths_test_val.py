import os
import json

# === Paths ===
metadata_path = "/thesis_main_files/data/metadata/lav-df/metadata.json"
test_dir = "/Users/abhishekgupte_macbookpro/Downloads/Datasets/LAV_DF/test"
dev_dir = "/Users/abhishekgupte_macbookpro/Downloads/Datasets/LAV_DF/dev"

# === Load metadata.json ===
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# === Lists to collect found and missing files ===
found_files = []
missing_files = []

# === Loop through metadata entries ===
for entry in metadata:
    file_path = entry['file']  # Example: "test/000001.mp4" or "dev/000015.mp4"

    if file_path.startswith('test/'):
        folder = test_dir
        relative_path = file_path.replace('test/', '')
    elif file_path.startswith('dev/'):
        folder = dev_dir
        relative_path = file_path.replace('dev/', '')
    else:
        continue  # Ignore unrelated entries

    full_path = os.path.join(folder, relative_path)

    if os.path.exists(full_path):
        found_files.append(full_path)
    else:
        missing_files.append(full_path)

# === Print Summary ===
print(f"\nTotal files checked: {len(found_files) + len(missing_files)}")
print(f"Files found: {len(found_files)}")
print(f"Files missing: {len(missing_files)}")

if missing_files:
    print("\nMissing files:")
    for missing in missing_files:
        print(missing)
else:
    print("\nNo missing files detected!")
