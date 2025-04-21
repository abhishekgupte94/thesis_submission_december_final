import torch
import tensorflow as tf
from tensorflow import keras
import mmcv

# Check PyTorch MPS support
print(f"PyTorch version: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Check TensorFlow GPU support
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

# Check Keras version
print(f"Keras version: {keras.__version__}")

# Check MMCV version
print(f"MMCV version: {mmcv.__version__}")
