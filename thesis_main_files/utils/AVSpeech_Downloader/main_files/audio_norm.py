import librosa
import os
import numpy as np
import scipy.io.wavfile as wavfile

# [NEW] Default audio index range used when running as a standalone script.
audio_range = (0, 20)

AUDIO_TRAIN_DIR = './audio_train'
NORM_AUDIO_DIR = './norm_audio_train'
SR = 16000


def normalize_file(src_path: str, dst_path: str, sr: int = SR) -> None:
    """[NEW] Normalize a single WAV file and write it to dst_path.

    This wraps the original audio_norm logic so it can be reused
    from other scripts (e.g., the Stage-1 downloader pipeline).
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source audio file not found: {src_path}")

    audio, _ = librosa.load(src_path, sr=sr)
    max_val = np.max(np.abs(audio)) if audio.size > 0 else 0.0

    if max_val > 0:
        norm_audio = np.divide(audio, max_val)
    else:
        # Edge case: silent or near-silent file; keep as-is
        norm_audio = audio

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    wavfile.write(dst_path, sr, norm_audio)


def main_batch():
    """[NEW] Preserve original batch behaviour over a fixed index range."""
    if not os.path.exists(NORM_AUDIO_DIR):
        os.mkdir(NORM_AUDIO_DIR)

    for idx in range(audio_range[0], audio_range[1]):
        print(f'Processing audio {idx}')
        path = os.path.join(AUDIO_TRAIN_DIR, f'trim_audio_train{idx}.wav')
        norm = os.path.join(NORM_AUDIO_DIR, f'trim_audio_train{idx}.wav')
        if os.path.exists(path):
            try:
                normalize_file(path, norm, sr=SR)
            except Exception as e:
                print(f'[audio_norm] Error normalizing index {idx}: {e}')


# if __name__ == '__main__':
#     main_batch()
