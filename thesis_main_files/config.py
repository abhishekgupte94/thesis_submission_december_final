# config.py

import torch
import platform

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")  # For Apple Silicon (M1/M2) with MPS backend
    else:
        return torch.device("cpu")

CONFIG = {
    "batch_size": 128,
    "learning_rate": 1e-4,
    "num_epochs": 20,
    "device": get_device(),
    "csv_name": "training_data_two.csv"
}