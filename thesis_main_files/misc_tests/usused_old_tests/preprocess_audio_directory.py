import os
# Function to delete .lab files in the specified directory and count files before and after
def delete_lab_files(audio_path):
    # Count files before deletion
    all_files_before = len([f for f in os.listdir(audio_path) if os.path.isfile(os.path.join(audio_path, f))])
    lab_files_before = len([f for f in os.listdir(audio_path) if f.endswith(".lab")])

    # Delete .lab files
    for root, _, files in os.walk(audio_path):
        for file in files:
            if file.endswith(".lab"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

    # Count files after deletion
    all_files_after = len([f for f in os.listdir(audio_path) if os.path.isfile(os.path.join(audio_path, f))])
    lab_files_after = len([f for f in os.listdir(audio_path) if f.endswith(".lab")])

    print(f"Total files before deletion: {all_files_before}, after deletion: {all_files_after}")
    print(f".lab files before deletion: {lab_files_before}, after deletion: {lab_files_after}")

# Delete .lab files in the specified audio directory
audio_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/data/processed_files/lav_df/audio_wav/train"
delete_lab_files(audio_path)


