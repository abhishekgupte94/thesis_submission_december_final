import h5py
import numpy as np
import os

os.environ.update({"KERAS_BACKEND": "tensorflow"})
from tensorflow.keras.models import load_model
from custom.model_modified import melody_ResNet_joint_add  # Ensure this is the updated model with 64 Mel bins

class Options:
    def __init__(self):
        self.num_spec = 64  # Updated for Mel-Spectrogram input (previously 513 for STFT)
        self.input_size = 31
        self.batch_size = 64
        self.resolution = 16
        self.figureON = False

options = Options()

# Load the existing JDC model with STFT-trained weights
old_model_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/melodyExtraction_JDC/weights/ResNet_joint_add_L(CE_G).hdf5"  # Path to STFT-based JDC model

# Load the new modified JDC model with Mel input shape
new_model = melody_ResNet_joint_add(options)

def fine_tune_jdc(old_model_path, new_model):
    """
    Transfers pretrained weights from STFT-based JDC to the Mel-Spectrogram-based JDC.
    Skips incompatible layers (e.g., input layers).
    Saves the fine-tuned model for future use.
    """
    with h5py.File(old_model_path, 'r') as f:
        for layer in new_model.layers:
            if layer.weights:
                try:
                    weights = [f[layer.name][w.name][()] for w in layer.weights]  # Updated for compatibility
                    layer.set_weights(weights)
                except:
                    print(f"⚠️ Skipping {layer.name}, shape mismatch.")  # Skip first layers due to input shape change

    # Save the fine-tuned model for later use
    new_model.save("fine_tuned_jdc_mel.hdf5")
    print("✅ Fine-tuned JDC model saved as 'fine_tuned_jdc_mel.hdf5'!")

    return new_model

# Run the weight transfer process
fine_tuned_model = fine_tune_jdc(old_model_path, new_model)
