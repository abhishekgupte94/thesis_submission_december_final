import re
from typing import Set

def count_unique_words_in_log(log_file_path: str) -> int:
    """
    Parse a .log file and count the number of unique words in it.

    Args:
        log_file_path (str): Path to the .log file.

    Returns:
        int: The count of unique words in the log file.
    """
    unique_words: Set[str] = set()

    try:
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()

        # Process each line to extract all unique words
        for line in lines:
            # Find all words in single quotes within the line
            matches = re.findall(r"'([^']+)'", line)
            unique_words.update(matches)

        # Print the unique words for debugging (optional)
        print(f"Unique words: {unique_words}")

        return len(unique_words)

    except FileNotFoundError:
        print(f"The file {log_file_path} was not found.")
        return 0
    except Exception as e:
        print(f"An error occurred while processing the log file: {e}")
        return 0


# Example usage
if __name__ == "__main__":
    mfa_log_file_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/utlis/log_files_mfa/mfa_oov.log"
    g2p_log_file_path  = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/utlis/log_files_mfa/possible_g2p_output.log"
    unique_word_count = count_unique_words_in_log(g2p_log_file_path)
    print(f"Total unique words: {unique_word_count}")
