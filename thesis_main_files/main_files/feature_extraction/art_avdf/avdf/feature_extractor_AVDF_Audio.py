import torch
import torchaudio
import torch.hub
import torch.nn as nn

class VGGishFeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

        # Enable multi-GPU support if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

    def extract_features(self, examples_tensor):
        """
        Extract features from a tensor input of shape [N, 1, 96, 64].
        This input is expected to be preprocessed in the same format as output from a DataLoader.
        """
        examples_tensor = examples_tensor.to(self.device)

        with torch.no_grad():
            embeddings = self.model(examples_tensor)  # Shape: [N, 128]

        return embeddings

    def extract_features_batch(self, batch_tensor):
        """
        Extract features from a batch of audio examples.
        batch_tensor should be a tensor of shape [B, 1, 96, 64] similar to a DataLoader batch.
        """
        return self.extract_features(batch_tensor)

# Example usage (from another class/module):
# extractor = VGGishFeatureExtractor()
# features = extractor.extract_features(batch_tensor)  # where batch_tensor is [B, 1, 96, 64]
