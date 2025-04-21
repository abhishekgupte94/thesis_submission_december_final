import os


def remove_corrupted_files(directory_path):
    """
    Identify and remove corrupted .wav files (0 KB) and their corresponding .lab files.

    Args:
        directory_path (str): Path to the directory containing .wav and .lab files.
    """
    # List to store the names of removed files
    removed_files = []

    # Iterate through all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.wav'):
            wav_file_path = os.path.join(directory_path, file_name)
            lab_file_path = os.path.join(directory_path, file_name.replace('.wav', '.lab'))

            # Check if the .wav file is 0 KB
            if os.path.exists(wav_file_path) and os.path.getsize(wav_file_path) == 0:
                # Remove the .wav file
                os.remove(wav_file_path)
                removed_files.append(wav_file_path)
                print(f"Removed corrupted .wav file: {wav_file_path}")

                # Check and remove the corresponding .lab file if it exists
                if os.path.exists(lab_file_path):
                    os.remove(lab_file_path)
                    removed_files.append(lab_file_path)
                    print(f"Removed corresponding .lab file: {lab_file_path}")

    # Print summary
    if removed_files:
        print("\nSummary of removed files:")
        for removed_file in removed_files:
            print(removed_file)
    else:
        print("No corrupted files found.")


# Example usage
if __name__ == "__main__":
    audio_directory = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/audio_wav/train_filenames"
    remove_corrupted_files(audio_directory)
