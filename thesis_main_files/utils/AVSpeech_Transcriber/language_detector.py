import whisper
import os

# ---------------------------------------------------------
# GLOBAL WHISPER LANGUAGE-ID MODEL
# ---------------------------------------------------------
_WHISPER_LID_MODEL = None

def load_whisper_lid_model():
    """
    Loads Whisper tiny once globally.
    The tiny model is ideal for fast & accurate language detection.
    """
    global _WHISPER_LID_MODEL
    if _WHISPER_LID_MODEL is None:
        _WHISPER_LID_MODEL = whisper.load_model("tiny")
    return _WHISPER_LID_MODEL


# ---------------------------------------------------------
# MAIN LANGUAGE DETECTION FUNCTION
# ---------------------------------------------------------
def detect_language_from_audio(audio_path: str) -> str:
    """
    Detects spoken language from a .wav (or any audio file)
    using Whisper’s automatic language identification.

    Parameters
    ----------
    audio_path : str
        Path to the pre-downloaded audio file (.wav or any format FFmpeg supports).

    Returns
    -------
    str
        ISO-639-1 language code (e.g., 'en', 'hi', 'ar', 'es').
        Returns None if detection fails.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    try:
        model = load_whisper_lid_model()

        # language=None triggers AUTODETECTION
        result = model.transcribe(
            audio_path,
            task="transcribe",
            language=None,
            verbose=False
        )

        lang = result.get("language")
        return lang

    except Exception as e:
        print(f"[Language Detection Error] {e}")
        return None


# ---------------------------------------------------------
# OPTIONAL: Batch processing utility
# ---------------------------------------------------------
def detect_languages_in_folder(folder_path: str):
    """
    Detects languages for all .wav files in a folder
    and returns a dict mapping filenames → detected language.
    """
    results = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            full_path = os.path.join(folder_path, filename)
            lang = detect_language_from_audio(full_path)
            results[filename] = lang
            print(f"{filename}: {lang}")

    return results




if __name__ == '__main__':
    lang = detect_language_from_audio("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/utils/AVSpeech_Downloader/main_files/audio_train/trim_audio_train1.wav")
    print(lang)  # -> 'en'
