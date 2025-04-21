import numpy as np
import os
import gc
from pathlib import Path


def load_valid_npy(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return None
    try:
        data = np.load(file_path)
        if not np.isfinite(data).all():
            print(f"⚠️ Skipping {file_path} due to invalid values!")
            return None
        return data
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")
        return None


def pad_and_concatenate(file_pairs):
    combined_batches = []
    for log_mel_path, melody_path in file_pairs:
        log_mel = load_valid_npy(log_mel_path)
        melody = load_valid_npy(melody_path)
        if log_mel is None or melody is None:
            continue
        Tl = log_mel.shape[2]
        Tm = melody.shape[1]
        melody = melody[:, np.newaxis, :]
        melody = np.repeat(melody, 64, axis=1)
        combined = np.concatenate((log_mel, melody), axis=2)
        combined_batches.append(combined)
    if not combined_batches:
        return None
    try:
        # Find the maximum T
        max_T = max(arr.shape[2] for arr in combined_batches)
        print(max_T," - Maximum length")
        # Pad each array along the third dimension (T)
        padded_arrays = [np.pad(arr, ((0, 0), (0, 0), (0, max_T - arr.shape[2])), mode='constant') for arr in combined_batches]

        # Stack the padded arrays
        return np.stack(padded_arrays, axis=0)

    except ValueError as e:
        print(f"❌ Shape mismatch error while stacking batches: {e}")
        return None
    del log_mel, melody
    gc.collect()
    return combined_batches


def get_batch_from_directories(log_mel_dir, melody_dir, num_real=5, num_fake=5):
    if not os.path.exists(log_mel_dir) or not os.path.exists(melody_dir):
        print("❌ One or both directories do not exist.")
        return None, None

    log_mel_files_real = sorted(
        f for f in os.listdir(os.path.join(log_mel_dir, "real")) if f.endswith("_log_mel_spec.npy"))
    log_mel_files_fake = sorted(
        f for f in os.listdir(os.path.join(log_mel_dir, "fake")) if f.endswith("_log_mel_spec.npy"))
    melody_files_real = sorted(f for f in os.listdir(os.path.join(melody_dir, "real")) if f.endswith("_melody.npy"))
    melody_files_fake = sorted(f for f in os.listdir(os.path.join(melody_dir, "fake")) if f.endswith("_melody.npy"))

    def match_files(log_mel_list, melody_list, base_dir_log, base_dir_mel, num_files):
        file_pairs = []
        for log_mel in log_mel_list:
            base_name = log_mel.split("_")[0]
            melody_file = f"{base_name}_melody.npy"
            if melody_file in melody_list:
                file_pairs.append((
                    os.path.join(base_dir_log, log_mel),
                    os.path.join(base_dir_mel, melody_file)
                ))
            if len(file_pairs) >= num_files:
                break
        return file_pairs[:num_files]

    real_pairs = match_files(log_mel_files_real, melody_files_real, os.path.join(log_mel_dir, "real"),
                             os.path.join(melody_dir, "real"), num_real)
    fake_pairs = match_files(log_mel_files_fake, melody_files_fake, os.path.join(log_mel_dir, "fake"),
                             os.path.join(melody_dir, "fake"), num_fake)

    real_batch = pad_and_concatenate(real_pairs) if real_pairs else None
    fake_batch = pad_and_concatenate(fake_pairs) if fake_pairs else None
    return real_batch, fake_batch


def save_batches(batch, output_dir, prefix="batch", batch_size=32):
    if batch is None or batch.shape[0] == 0:
        print("❌ No valid batch to save.")
        return
    os.makedirs(output_dir, exist_ok=True)
    num_batches = max(1, (batch.shape[0] + batch_size - 1) // batch_size)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, batch.shape[0])
        batch_slice = batch[start_idx:end_idx]
        file_path = os.path.join(output_dir, f"{prefix}_{i + 1}.npy")
        np.save(file_path, batch_slice)
        print(f"✅ Saved {file_path} with shape {batch_slice.shape}")


# Example Usage
log_mel_dir = Path(
    __file__).resolve().parent.parent / "datasets" / "processed" / "lav_df" / "audio_wav" / "audio_spectograms"
melody_dir = Path(__file__).resolve().parent.parent / "datasets" / "processed" / "lav_df" / "audio_wav" / "melody"
output_dir = Path(__file__).resolve().parent.parent / "datasets" / "processed" / "lav_df" / "audio_wav" / "concatenated"
batch_size = 1  # Set batch size dynamically
num_real = 20
num_fake = 20
real_batch, fake_batch = get_batch_from_directories(log_mel_dir, melody_dir, num_real=num_real, num_fake=num_fake)

save_batches(real_batch, output_dir, prefix="real_batch", batch_size=batch_size)
save_batches(fake_batch, output_dir, prefix="fake_batch", batch_size=batch_size)
