"""
Parallel Whisper batch transcription (CPU-only) with word-level timestamps and logging.

- Hardcoded paths / settings.
- CPU-only: forces Whisper to run on CPU.
- Parallelised over files with a small process pool (3–4 workers).
- Each worker process loads its own Whisper model ONCE.
- Parent process manages file cap + log file.
- [ADDED] A global CSV index that stores each audio file name + [[start, end], ...] timestamps.
"""

import os
import csv
import time
import gc  # [MEM] to manually trigger garbage collection after large objects
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import whisper
import torch  # for explicit device control
import json   # [ADDED] for serialising the [[s,e], ...] list


# ========================== USER CONFIG ==========================

AUDIO_DIR = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/video_files/AVSpeech/audio"

# [MODIFIED] OUT_DIR should be a **directory**, not a single .csv file.
OUT_DIR = "/data/processed/AVSpeech/AVSpeech_timestamps_csv"

LOG_FILE = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/temp_files/AVSpeech/transcribed_files.txt"

WHISPER_MODEL_NAME = "tiny"    # "tiny" or "base" are good for CPU; "small" will be slower
MAX_NEW_FILES = 20000            # Cap on newly processed files this run
NUM_WORKERS = 3                # You requested 3 workers (CPU-only)

# [ADDED] Global CSV index to store file name + timestamps list [[s1,e1],[s2,e2],...]
GLOBAL_INDEX_CSV = os.path.join(
    OUT_DIR,
    "AVSpeech_timestamps_index.csv"
)

# ================================================================


# NOTE:
# Previously we had transcribe_with_timestamps_local() that built large Python lists
# and returned them. That keeps all word dicts in memory at once.
#
# Now we *stream* the CSV writing directly from the Whisper result to reduce peak RAM.


def save_word_timestamps_to_csv_local(model, audio_path: str, out_csv_path: str):
    """
    Transcribe with given model (CPU, fp16 disabled on CPU) and
    save word timestamps to CSV.

    IMPLEMENTATION NOTES:
    - Uses fp16=False to avoid "floating point 16 is not supported on CPU" warnings/errors.
    - Writes rows to CSV as we iterate over segments/words (no big in-memory word list).
    - Cleans up large 'result' object explicitly.
    """
    # --- Transcribe (CPU, float32) ---
    result = model.transcribe(
        audio_path,
        task="transcribe",
        language=None,         # auto-detect
        word_timestamps=True,
        verbose=False,
        fp16=False,            # [FIX] Force fp32 on CPU
    )

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    # --- Stream-write to CSV as we iterate ---
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_file", "segment_index", "word", "start", "end"])

        for i, seg in enumerate(result.get("segments", [])):
            words = seg.get("words", []) or []
            for w in words:
                word_txt = (w.get("word") or "").strip()
                start = float(w.get("start", 0.0))
                end = float(w.get("end", 0.0))

                writer.writerow([
                    os.path.basename(audio_path),
                    i,
                    word_txt,
                    f"{start:.3f}",
                    f"{end:.3f}",
                ])

        # optional explicit flush, though closing the file handles this
        f.flush()

    # --- Free large objects ASAP ---
    del result
    gc.collect()


# ------------------ LOG HANDLING (parent process) ------------------

def load_transcribed_log(log_file: str) -> set:
    if not os.path.exists(log_file):
        return set()

    processed = set()
    # [ROBUST] Guard against weird log corruption / unreadable lines
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    processed.add(name)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Could not fully read log file '{log_file}': {e}")
    return processed


def append_to_transcribed_log(log_file: str, filename: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    try:  # [ROBUST]
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(filename + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed to append '{filename}' to log '{log_file}': {e}")


# ------------------ NEW HELPERS FOR GLOBAL CSV INDEX (parent) ------------------

def load_word_pairs_from_perfile_csv(perfile_csv_path: str) -> List[List[float]]:
    """
    Read a per-file *_words.csv and return [[start, end], ...] in float seconds,
    one entry per word (already 'divided by words').
    """
    pairs: List[List[float]] = []
    if not os.path.exists(perfile_csv_path):
        return pairs

    # [ROBUST] Protect against partial/invalid CSV
    try:
        with open(perfile_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    start = float(row["start"])
                    end = float(row["end"])
                except (KeyError, ValueError, TypeError):  # noqa: PERF203
                    # Skip malformed rows but do not crash the whole run
                    continue
                pairs.append([start, end])
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed reading per-file CSV '{perfile_csv_path}': {e}")

    return pairs


def append_to_global_index(global_csv_path: str, audio_file: str, word_pairs: List[List[float]]):
    """
    Append one row to the global index CSV:

        audio_file, timestamps_pairs

    where timestamps_pairs is a JSON string like:
        [[0.32, 0.55], [0.55, 0.80], ...]
    """
    os.makedirs(os.path.dirname(global_csv_path), exist_ok=True)

    file_exists = os.path.exists(global_csv_path)

    # [ROBUST] Protect against I/O issues when updating global index
    try:
        with open(global_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["audio_file", "timestamps_pairs"])

            timestamps_str = json.dumps(word_pairs, ensure_ascii=False)
            writer.writerow([audio_file, timestamps_str])
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed appending to global index '{global_csv_path}' for '{audio_file}': {e}")


# ------------------ WORKER FUNCTION (CPU-only) ------------------

def worker_transcribe_one(args: Tuple[str, str, str, str]) -> Tuple[str, bool, str]:
    """
    Worker entry point (CPU-only).

    Parameters (packed in args):
        audio_dir : str
        filename  : str
        out_dir   : str
        model_name: str

    Returns
    -------
    (filename, success_flag, error_message)
    """
    audio_dir, filename, out_dir, model_name = args

    full_path = os.path.join(audio_dir, filename)

    # [ROBUST] Skip missing files cleanly
    if not os.path.exists(full_path):
        return filename, False, f"Audio file not found: {full_path}"

    # Each worker keeps its own global model instance (on CPU).
    global _WORKER_MODEL  # type: ignore[name-defined]
    try:
        _WORKER_MODEL  # type: ignore[name-defined]
    except NameError:
        try:  # [ROBUST] Guard against model load failures
            print(f"[WORKER {os.getpid()}] Loading model on CPU: {model_name}")
            _WORKER_MODEL = whisper.load_model(model_name, device="cpu")  # type: ignore[assignment]
            _WORKER_MODEL.to("cpu")  # type: ignore[attr-defined]
            torch.set_num_threads(1)  # prevent each worker from oversubscribing CPU threads
        except Exception as e:  # noqa: BLE001
            return filename, False, f"Failed to load model: {e}"

    base = os.path.splitext(filename)[0]
    out_csv = os.path.join(out_dir, base + "_words.csv")

    try:
        save_word_timestamps_to_csv_local(_WORKER_MODEL, full_path, out_csv)  # type: ignore[arg-type]
        # [MEM] Light clean-up per file
        gc.collect()
        return filename, True, ""
    except Exception as e:  # noqa: BLE001
        # [ROBUST] If transcription fails, skip this file but don't kill the pool
        gc.collect()
        return filename, False, str(e)


# ------------------ BATCH CONTROLLER (parent) ------------------

def batch_transcribe_parallel_cpu():
    os.makedirs(OUT_DIR, exist_ok=True)

    already_processed = load_transcribed_log(LOG_FILE)
    print(f"[INFO] Already processed (from log): {len(already_processed)}")

    # [ROBUST] Guard against problems listing AUDIO_DIR
    try:
        all_files = sorted([
            f for f in os.listdir(AUDIO_DIR)
            if f.lower().endswith(".wav")
        ])
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Could not list audio directory '{AUDIO_DIR}': {e}")
        return

    # Filter out those already done
    files_to_do = [f for f in all_files if f not in already_processed]

    if not files_to_do:
        print("[INFO] No new files to process.")
        return

    # Apply cap
    files_to_do = files_to_do[:MAX_NEW_FILES]
    print(f"[INFO] Will process up to {len(files_to_do)} new files this run "
          f"(cap={MAX_NEW_FILES}, workers={NUM_WORKERS}, CPU-only).")

    start_time = time.time()
    completed = 0

    # Build args for workers
    jobs = [
        (AUDIO_DIR, filename, OUT_DIR, WHISPER_MODEL_NAME)
        for filename in files_to_do
    ]

    # Parallel processing with CPU workers
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_file = {executor.submit(worker_transcribe_one, job): job[1] for job in jobs}

        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                fname, success, err_msg = future.result()
            except Exception as e:  # noqa: BLE001
                # [ROBUST] Catch catastrophic worker failures and continue
                print(f"[ERROR] {filename}: worker crashed with {e}")
                continue

            if success:
                print(f"[DONE] {fname}")
                append_to_transcribed_log(LOG_FILE, fname)
                completed += 1

                # [ADDED] Build the per-file path and append to global index.
                base = os.path.splitext(fname)[0]
                perfile_csv = os.path.join(OUT_DIR, base + "_words.csv")

                # [ROBUST] Make global index update best-effort only
                try:
                    word_pairs = load_word_pairs_from_perfile_csv(perfile_csv)
                    append_to_global_index(GLOBAL_INDEX_CSV, fname, word_pairs)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] Failed to update global index for '{fname}': {e}")

            else:
                # [ROBUST] Log error and skip file
                print(f"[ERROR] {fname}: {err_msg}")

    elapsed = time.time() - start_time
    print(f"[INFO] Completed {completed} files in {elapsed:.2f} seconds.")


# ------------------ MAIN ------------------

if __name__ == "__main__":
    batch_transcribe_parallel_cpu()
