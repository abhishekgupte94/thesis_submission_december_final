import torch

# Create a tensor on the CPU
x = torch.randn(3, 3)

# Move the tensor to the GPU
if torch.cuda.is_available():
    print("GPU support exists on this machine!")
else:
    print("GPU support does not exist on this machine!")
