from extract_audio_features_from_AST import ASTAudioExtractor

# pick device depending on your env (cuda:0 on your A100 box, cpu on laptop)
extractor = ASTAudioExtractor(
    device="cpu",   # or "cpu" if you don't have GPU locally
    amp=True,
    time_series=True,  # default
    verbose=True
)
print("✅ extractor initialized")



# test with a known video that has audio track
video_path = "/thesis_main_files/datasets/processed/lav_df/new_setup/evaluate_files/A_only_lt7p5/000035.mp4"

res = extractor.extract_one(video_path, save=False)
print("Keys:", res.keys())
print("Shape:", res["shape"])
print("Features tensor type:", type(res["features"]))
print("First 2 timesteps:", res["features"][:2])

