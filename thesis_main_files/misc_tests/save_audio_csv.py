import os
import pandas as pd
# import multiprocessing
# import moviepy.editor as mp
import json


def convert_to_wav_format(input_csv_path, output_csv_path):
    """
    Convert file names in the input CSV to .wav format and save them to a new CSV.
    Args:
        input_csv_path (str): Path to the input CSV file containing file names and transcripts.
        output_csv_path (str): Path to save the updated CSV file with .wav file names.
    """
    try:
        # Load input CSV
        df = pd.read_csv(input_csv_path)

        # Modify file names to .wav format
        df['file_wav'] = df['file'].apply(lambda x: os.path.basename(x).replace('.mp4', '.wav'))
        df = df.drop(columns=['file'])

# Convert transcripts to '"str"' format
        df['transcript'] = df['transcript'].apply(lambda x: f'"{x}"')

# Reorder columns so 'file_wav' is the first column
        df = df[['file_wav', 'transcript']]

        # Save updated data to a new CSV
        df.to_csv(output_csv_path, index=False)
        print(f"Updated transcripts saved to: {output_csv_path}")
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except pd.errors.EmptyDataError as empty_error:
        print(f"The CSV file {input_csv_path} is empty or improperly formatted: {empty_error}")
    except Exception as e:
        print(f"Error processing CSV file: {e}")



if __name__ == "__main__":
    transcripts_csv_path = "/datasets/processed/csv_files/lav_df/transcripts/transcripts.csv"

    updated_transcripts_csv_path = "/datasets/processed/csv_files/lav_df/transcripts/audio_against_transcripts.csv"
    convert_to_wav_format(transcripts_csv_path, updated_transcripts_csv_path)
