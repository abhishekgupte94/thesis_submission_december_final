# sanity_audio_preprocessor_npv.py

from __future__ import annotations

from pathlib import Path
import time
import torch

from audio_preprocessor_npv import AudioPreprocessorNPV, AudioPreprocessorConfig


def main() -> None:
    t0 = time.time()

    # ----------------------------
    # EDIT THESE PATHS
    # ----------------------------
    audio_path = Path("/ABS/PATH/TO/YOUR_AUDIO.wav")
    out_pt_path = Path("/ABS/PATH/TO/OUTPUT/audio_segments.pt")

    # In-memory timestamps (seconds)
    # Format: [[start_sec, end_sec], ...]
    word_times = [
        [0.10, 0.35],
        [0.40, 0.80],
        [1.20, 1.55],
        [1.60, 2.05],
    ]

    cfg = AudioPreprocessorConfig(
        target_sr=16000,
        n_mels=80,
        target_num_frames=250,
    )
    proc = AudioPreprocessorNPV(cfg)

    num_segments, num_words = proc.process_and_save_from_timestamps_csv_segmentlocal(
        audio_path=audio_path,
        word_times=word_times,
        out_pt_path=out_pt_path,
        log_csv_path=None,
    )

    payload = torch.load(out_pt_path, map_location="cpu")

    # ==================================================================
    # [MODIFIED] Minimal payload assertions: ONLY these keys must exist
    # ==================================================================
    assert set(payload.keys()) == {"audio_file", "mel_segments", "segments_sec"}, (
        f"Payload keys mismatch. Got: {sorted(payload.keys())}"
    )

    assert payload["audio_file"] == audio_path.name

    mel_segments = payload["mel_segments"]
    segments_sec = payload["segments_sec"]

    assert isinstance(mel_segments, list), "mel_segments should be a list"
    assert isinstance(segments_sec, list), "segments_sec should be a list"
    assert len(mel_segments) == len(segments_sec) == num_segments

    for i, mel in enumerate(mel_segments):
        assert isinstance(mel, torch.Tensor), f"mel_segments[{i}] not a Tensor"
        assert mel.ndim == 2, f"mel_segments[{i}] expected 2D (n_mels, T), got {tuple(mel.shape)}"
        assert mel.shape[0] == cfg.n_mels, f"mel_segments[{i}] n_mels mismatch"
        assert mel.shape[1] == cfg.target_num_frames, f"mel_segments[{i}] T mismatch"
        assert mel.dtype == torch.float32, f"mel_segments[{i}] dtype mismatch"

    for i, (s, e) in enumerate(segments_sec):
        assert isinstance(s, float) and isinstance(e, float), f"segments_sec[{i}] not float tuple"
        assert e > s, f"segments_sec[{i}] invalid: {(s, e)}"

    dt = time.time() - t0
    print("=== [SANITY] AudioPreprocessorNPV (minimal payload) ===")
    print(f"audio: {audio_path}")
    print(f"saved: {out_pt_path}")
    print(f"num_words={num_words} num_segments={num_segments}")
    print(f"elapsed_sec={dt:.2f}")
    print("OK")


if __name__ == "__main__":
    main()
