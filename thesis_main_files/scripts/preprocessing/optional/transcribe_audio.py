import whisper

model = whisper.load_model("medium")   # or "small", "large", etc.

result = model.transcribe(
    "your_file.mp4",
    word_timestamps=True,   # requires a recent version
    verbose=False
)

# Segments with word-level timestamps
for seg in result["segments"]:
    print(f"[{seg['start']:.2f} -> {seg['end']:.2f}] {seg['text']}")
    if "words" in seg:  # word-level
        for w in seg["words"]:
            print(f"   {w['word']}  ({w['start']:.2f}–{w['end']:.2f})")
