# from lingua import Language, LanguageDetectorBuilder
# import pandas as pd
#
#
# def scan_for_english(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
#     # Initialize detector with minimal language set for efficiency
#     detector = LanguageDetectorBuilder.from_languages(
#         Language.ENGLISH,
#         Language.FRENCH  # Required second language for comparison
#     ).with_minimum_relative_distance(0.25).build()
#
#     def is_english(text: str) -> bool:
#         try:
#             if pd.isna(text) or text.strip() == '':
#                 return False
#             detected = detector.detect_language_of(text)
#             return detected == Language.ENGLISH
#         except:
#             return False
#
#     # Add English detection column
#     df['is_english'] = df[text_column].apply(is_english)
#
#     # Print summary statistics
#     total_rows = len(df)
#     english_rows = df['is_english'].sum()
#     print(f"Total rows: {total_rows}")
#     print(f"English rows: {english_rows}")
#     print(f"Percentage English: {(english_rows / total_rows) * 100:.2f}%")
#     print(df['is_english'])
#     return 0
#
#
# def scan_for_non_english(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
#     # Initialize detector with minimal language set for efficiency
#     detector = LanguageDetectorBuilder.from_languages(
#         Language.ENGLISH,
#         Language.FRENCH  # Required second language for comparison
#     ).with_minimum_relative_distance(0.25).build()
#
#     def is_english(text: str) -> bool:
#         try:
#             if pd.isna(text) or text.strip() == '':
#                 return False
#             detected = detector.detect_language_of(text)
#             return detected != Language.ENGLISH
#         except:
#             return False
#
#     # Add English detection column
#     df['not_english'] = df[text_column].apply(is_english)
#
#     # Print summary statistics
#     total_rows = len(df)
#     english_rows = df['not_english'].sum()
#     print(f"Total rows: {total_rows}")
#     print(f"Non-English rows: {english_rows}")
#     print(f"Percentage Non-English: {(english_rows / total_rows) * 100:.2f}%")
#     print(df['not_english'])
#     df_non_english = df['not_english'].copy()
#     df_non_english = df_non_english.file
#     df_non_english.to_csv("/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/csv_files/lav_df/non_english_transcripts.csv")
#     return 0
#
# # Example usage
# transcripts_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/csv_files/lav_df/transcripts_train.csv"
# # save_path = "/datasets/processed/csv_files/lav_df/transcripts/video_against_transcripts.csv"
#
# df = pd.read_csv(transcripts_path)
# df['not_english'] = scan_for_non_english(df, 'transcript')
# print(df['not_english'])
#
#
# def clean_non_english(df: pd.DataFrame, text_column: str, output_file: str) -> pd.DataFrame:
#     # Initialize detector
#     detector = LanguageDetectorBuilder.from_languages(
#         Language.ENGLISH,
#         Language.FRENCH
#     ).with_minimum_relative_distance(0.25).build()
#
#     def is_english(text: str) -> bool:
#         try:
#             if pd.isna(text) or text.strip() == '':
#                 return False
#             detected = detector.detect_language_of(text)
#             return detected == Language.ENGLISH
#         except:
#             return False
#
#     # Add language detection column
#     df['is_english'] = df[text_column].apply(is_english)
#
#     # Keep only English rows
#     english_df = df[df['is_english']].copy()
#
#     # Remove the temporary is_english column
#     english_df = english_df.drop('is_english', axis=1)
#
#     # Save to CSV
#     english_df.to_csv(output_file, index=False)
#
#     # Print summary
#     total_rows = len(df)
#     english_rows = len(english_df)
#     removed_rows = total_rows - english_rows
#     print(f"Original rows: {total_rows}")
#     print(f"Rows kept (English): {english_rows}")
#     print(f"Rows removed (non-English): {removed_rows}")
#     print(f"Percentage kept: {(english_rows / total_rows) * 100:.2f}%")
#     print(f"\nSaved clean data to: {output_file}")
#
#     return english_df
#
#
# # Usage
# # cleaned_df = clean_non_english(df, 'transcript', save_path)


from lingua import Language, LanguageDetectorBuilder
import pandas as pd
import os


def scan_for_non_english(df: pd.DataFrame, text_column: str, output_file: str) -> None:
    """
    Scan for non-English rows and save their file names as a CSV.

    Args:
        df (pd.DataFrame): The input dataframe.
        text_column (str): The column name containing text to check.
        output_file (str): Path to save non-English file names.
    """
    # Initialize detector
    detector = LanguageDetectorBuilder.from_languages(
        Language.ENGLISH,
        Language.FRENCH  # Required second language for comparison
    ).with_minimum_relative_distance(0.25).build()

    def is_non_english(text: str) -> bool:
        try:
            if pd.isna(text) or text.strip() == '':
                return False
            detected = detector.detect_language_of(text)
            return detected != Language.ENGLISH
        except:
            return False

    # Add non-English detection column
    df['not_english'] = df[text_column].apply(is_non_english)

    # Filter non-English rows
    non_english_df = df[df['not_english']].copy()

    # Save 'file' column to CSV
    non_english_df[['file']].to_csv(output_file, index=False)
    print(non_english_df)
    print(f"Non-English file names saved to: {output_file}")


def delete_non_english_files(file_list_csv: str, directory_path: str) -> None:
    """
    Delete .wav and .lab files listed in the CSV from the specified directory.

    Args:
        file_list_csv (str): Path to the CSV file containing file names to delete.
        directory_path (str): Directory containing the .wav and .lab files to delete.
    """
    try:
        # Read the file list CSV
        file_list_df = pd.read_csv(file_list_csv)
        files_to_delete = file_list_df['file'].tolist()

        for file_name in files_to_delete:
            # Derive the .wav and .lab file paths
            wav_file = os.path.join(directory_path, file_name.replace('.mp4', '.wav'))
            lab_file = os.path.join(directory_path, file_name.replace('.mp4', '.lab'))

            # Delete .wav file
            if os.path.exists(wav_file):
                os.remove(wav_file)
                print(f"Deleted: {wav_file}")
            else:
                print(f"File not found, skipping: {wav_file}")

            # Delete .lab file
            if os.path.exists(lab_file):
                os.remove(lab_file)
                print(f"Deleted: {lab_file}")
            else:
                print(f"File not found, skipping: {lab_file}")
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"Error deleting files: {e}")


# Example usage
if __name__ == "__main__":
    transcripts_csv_path = "/datasets/processed/csv_files/lav_df/transcripts_train.csv"
    non_english_output_file = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/misc_files/non_english_files_to_delete.csv"
    audio_directory = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/datasets/processed/lav_df/audio_wav/train_filenames"

    # Load transcripts CSV
    transcripts_df = pd.read_csv(transcripts_csv_path)

    # Scan for non-English files and save to CSV
    # scan_for_non_english(transcripts_df, 'transcript', non_english_output_file)

    # Delete corresponding .wav and .lab files
    delete_non_english_files(non_english_output_file, audio_directory)
