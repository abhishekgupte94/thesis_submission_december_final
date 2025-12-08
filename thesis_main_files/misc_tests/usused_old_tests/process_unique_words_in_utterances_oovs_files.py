
def count_unique_words_in_file(file_path):
    """
    Count the unique words in a file.

    Args:
        file_path (str): Path to the file to analyze.

    Returns:
        int: Total number of unique words.
    """
    with open(file_path, "r") as file:
        words = set(word.strip().lower() for word in file if word.strip())
    print(f"Unique words in {file_path}: {len(words)}")
    return len(words)


def compare_oovs(utterance_file, arpa_file):
    """
    Compare words from utterance_oovs.txt against oovs_found_english_us_arpa.txt.

    Args:
        utterance_file (str): Path to the utterance_oovs.txt file.
        arpa_file (str): Path to the oovs_found_english_us_arpa.txt file.

    Returns:
        tuple: A list of matches found in the utterance file, total matched words in ARPA.
    """
    def parse_utterance_line(line):
        # Extract the portion after the first colon
        parts = line.split(":")
        if len(parts) < 2:
            return []
        utterances = parts[1].strip()
        # Split utterances into words, separating by ", " or similar patterns
        words = [
            "".join(word.split(", ")).replace(" ", "") for word in utterances.split(", ")
        ]
        return words

    # Load ARPA words
    with open(arpa_file, "r") as arpa_f:
        arpa_words = set(word.strip().lower() for word in arpa_f if word.strip())

    matches_in_utterances = []
    matched_arpa_words = set()
    total_utterance_matches = 0

    # Parse utterance file and compare
    with open(utterance_file, "r") as utterance_f:
        for line in utterance_f:
            utterance_words = parse_utterance_line(line)
            for word in utterance_words:
                if word.lower() in arpa_words:
                    matches_in_utterances.append(word)
                    matched_arpa_words.add(word.lower())
                    total_utterance_matches += 1

    # Count matched words in ARPA file
    total_matched_arpa_words = len(matched_arpa_words)

    # Print summary
    print(f"Matches found in utterances: {total_utterance_matches}")
    print(f"Unique matched words in ARPA file: {total_matched_arpa_words}")
    print("Matched words:", matches_in_utterances)

    return matches_in_utterances, total_matched_arpa_words
def count_and_print_unique_words_in_file(file_path):
    """
    Count and print the unique words in a file.

    Args:
        file_path (str): Path to the file to analyze.

    Returns:
        set: A set of unique words.
    """
    with open(file_path, "r") as file:
        unique_words = set(word.strip().lower() for word in file if word.strip())
    print(f"Unique words in {file_path}: {len(unique_words)}")
    print("Words:", unique_words)
    return unique_words



# Example usage
if __name__ == "__main__":
    utterance_oovs_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/utlis/oovs_and_utterances/utterance_oovs.txt"
    arpa_words_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/utlis/oovs_and_utterances/oovs_found_english_us_arpa.txt"
    #

    # Count unique words in each file
    # count_unique_words_in_file(utterance_oovs_path)
    # count_unique_words_in_file(arpa_words_path)
    count_and_print_unique_words_in_file(utterance_oovs_path)
    # count_and_print_unique_words_in_file(arpa_words_path)
    # Compare the files
    matches, matched_arpa_word_count = compare_oovs(utterance_oovs_path, arpa_words_path)
