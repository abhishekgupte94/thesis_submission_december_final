from lingua import Language, LanguageDetectorBuilder
import pandas as pd


def scan_for_english(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    # Initialize detector with minimal language set for efficiency
    detector = LanguageDetectorBuilder.from_languages(
        Language.ENGLISH,
        Language.FRENCH  # Required second language for comparison
    ).with_minimum_relative_distance(0.25).build()

    def is_english(text: str) -> bool:
        try:
            if pd.isna(text) or text.strip() == '':
                return False
            detected = detector.detect_language_of(text)
            return detected == Language.ENGLISH
        except:
            return False

    # Add English detection column
    df['is_english'] = df[text_column].apply(is_english)

    # Print summary statistics
    total_rows = len(df)
    english_rows = df['is_english'].sum()
    print(f"Total rows: {total_rows}")
    print(f"English rows: {english_rows}")
    print(f"Percentage English: {(english_rows / total_rows) * 100:.2f}%")
    print(df['is_english'])
    return 0

# Example usage
transcripts_path = "/datasets/processed/csv_files/lav_df/transcripts/transcripts.csv"
df = pd.read_csv(transcripts_path)
df['is_english'] = scan_for_english(df, 'transcript')
# print(df['is_english'])
