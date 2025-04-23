from pprint import pprint
import torch
from thesis_main_files.models.svm.create_data_loader_for_svm import FeatureBuilder
from thesis_main_files.models.svm.create_data_loader_for_inference import AudioVideoInferenceDataset
from thesis_main_files.models.svm.train_svm_on_deep_features import SVMTrainer
from thesis_main_files.models.svm.evaluate_svm import SVMEvaluator
# (1) Simulated input deep features and labels
audio_tensor = torch.randn(100, 128)
video_tensor = torch.randn(100, 128)
labels_tensor = torch.randint(0, 2, (100,))

# (2) Create the dataset
dataset = AudioVideoInferenceDataset(audio_tensor, video_tensor, labels_tensor)

# (3) Define your model inference function
def dummy_model_inference(audio_input, video_input):
    return audio_input * 2, video_input * 3

# (4) Binary label conversion
binary_fn = lambda x: int(x == 1)

# (5) Extract concatenated features
deep_dataset = FeatureBuilder.build_dataset(dataset, dummy_model_inference, binary_fn)

# (6) Train the SVM
svm_model, X_test, y_test = SVMTrainer.train(deep_dataset)

# (7) Evaluate it
metrics = SVMEvaluator.evaluate(svm_model, X_test, y_test)
pprint(metrics)
