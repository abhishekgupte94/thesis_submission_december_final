import pandas as pd

import pandas as pd
import os
file_path = "/datasets/processed/csv_files/lav_df/training_data/training_data_two.csv"


def save_filenames(csv_file, output_txt="filenames.txt", num_files=20):
    """
    Extracts the filenames from the 'video_path' column of a CSV file and saves them to a .txt file.
    Each line in the text file contains: <filename> 0

    Parameters:
        csv_file (str): Path to the CSV file.
        output_txt (str): Path to the output .txt file (default: 'filenames.txt').
        num_files (int): Number of filenames to extract (default: 20).

    Returns:
        None
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Check if 'video_path' column exists
        if 'video_path' not in df.columns:
            raise ValueError("Column 'video_path' not found in the CSV file.")

        # Extract the specified number of filenames
        filenames = df['video_path'].dropna().astype(str).apply(os.path.basename)[:num_files]

        # Save to .txt file
        with open(output_txt, "w") as f:
            for filename in filenames:
                f.write(f"{filename} 0\n")

        print(f"Filenames saved to {output_txt}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
# save_filenames("your_file.csv", "output.txt", 50)  # Extracts first 50 filenames

output_path = "/data/train/lip_train_text.txt"
# Example usage
save_filenames(file_path,output_path)

