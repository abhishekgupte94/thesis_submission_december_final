"""
Parallel Whisper batch transcription (CPU-only) with word-level timestamps and logging.

- Hardcoded paths / settings.
- CPU-only: forces Whisper to run on CPU.
- Parallelised over files with a small process pool (3 workers).
- Each worker process loads its own Whisper model ONCE.
- Parent process manages file cap + log file.
"""

import os
import csv
import time
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import whisper
import torch  # for explicit device control


# ========================== USER CONFIG ==========================

AUDIO_DIR = "//Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/video_files/AVSpeech/audio"
OUT_DIR = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/AVSpeech/AVSpeech_timestamps.csv"
LOG_FILE = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/temp_files/AVSpeech/transcribed_files.txt"

WHISPER_MODEL_NAME = "tiny"    # "tiny" or "base" are good for CPU; "small" will be slower
MAX_NEW_FILES = 100              # Cap on newly processed files this run
NUM_WORKERS = 4                # You requested 3 workers (CPU-only)

# ================================================================


def transcribe_with_timestamps_local(model, audio_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transcribe a single audio file with a given Whisper model instance (CPU),
    returning segments + word timestamps.
    """
    result = model.transcribe(
        audio_path,
        task="transcribe",
        language=None,        # auto-detect
        word_timestamps=True,
        verbose=False
    )

    segments_out: List[Dict[str, Any]] = []
    words_out: List[Dict[str, Any]] = []

    for i, seg in enumerate(result.get("segments", [])):
        segments_out.append({
            "index": i,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg.get("text", "").strip(),
        })

        for w in seg.get("words", []):
            words_out.append({
                "segment_index": i,
                "word": w["word"].strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
            })

    return {
        "segments": segments_out,
        "words": words_out,
    }


def save_word_timestamps_to_csv_local(model, audio_path: str, out_csv_path: str):
    """
    Transcribe with given model (CPU) and save word timestamps to CSV.
    (Runs inside worker processes.)
    """
    data = transcribe_with_timestamps_local(model, audio_path)
    words = data["words"]

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_file", "segment_index", "word", "start", "end"])

        for w in words:
            writer.writerow([
                os.path.basename(audio_path),
                w["segment_index"],
                w["word"],
                f"{w['start']:.3f}",
                f"{w['end']:.3f}",
            ])


# ------------------ LOG HANDLING (parent process) ------------------

def load_transcribed_log(log_file: str) -> set:
    if not os.path.exists(log_file):
        return set()

    processed = set()
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                processed.add(name)
    return processed


def append_to_transcribed_log(log_file: str, filename: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


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

    # Each worker keeps its own global model instance (on CPU).
    global _WORKER_MODEL
    try:
        _WORKER_MODEL
    except NameError:
        print(f"[WORKER {os.getpid()}] Loading model on CPU: {model_name}")
        # Explicitly load on CPU
        _WORKER_MODEL = whisper.load_model(model_name, device="cpu")
        # Just to be sure:
        _WORKER_MODEL.to("cpu")
        torch.set_num_threads(1)  # prevent each worker from oversubscribing CPU threads

    full_path = os.path.join(audio_dir, filename)
    base = os.path.splitext(filename)[0]
    out_csv = os.path.join(out_dir, base + "_words.csv")

    try:
        save_word_timestamps_to_csv_local(_WORKER_MODEL, full_path, out_csv)
        return filename, True, ""
    except Exception as e:
        return filename, False, str(e)


# ------------------ BATCH CONTROLLER (parent) ------------------

def batch_transcribe_parallel_cpu():
    os.makedirs(OUT_DIR, exist_ok=True)

    already_processed = load_transcribed_log(LOG_FILE)
    print(f"[INFO] Already processed (from log): {len(already_processed)}")

    # Get .wav list
    all_files = sorted([
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".wav")
    ])

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
            except Exception as e:
                print(f"[ERROR] {filename}: worker crashed with {e}")
                continue

            if success:
                print(f"[DONE] {fname}")
                append_to_transcribed_log(LOG_FILE, fname)
                completed += 1
            else:
                print(f"[ERROR] {fname}: {err_msg}")

    elapsed = time.time() - start_time
    print(f"[INFO] Completed {completed} files in {elapsed:.2f} seconds.")


# ------------------ MAIN ------------------

if __name__ == "__main__":
    batch_transcribe_parallel_cpu()
