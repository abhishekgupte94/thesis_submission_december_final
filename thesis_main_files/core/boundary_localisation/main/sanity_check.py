from core.boundary_localisation.main.build_model import BATFDInferenceWrapper

model = BATFDInferenceWrapper(
    checkpoint_path="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/pretrained_weights/batfd_plus_default.ckpt",
    model_type="batfd_plus",
    device="mps",
)

video, audio = BATFDInferenceWrapper.preprocess_from_paths("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/splits/train/video/000470.mp4")
print("[INFO]", video.shape, audio.shape)

out = model(video, audio)
print("OK forward")
