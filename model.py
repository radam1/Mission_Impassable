"""
model.py: 1.0
This file contains the model architecture for processing visual input into the specific corner points for the drone
"""
import torch 


print(torch.backends.mps.is_available())  # Should return True on M2 Max

# Check if PyTorch can use the MPS (Metal Performance Shaders) backend
print(torch.backends.mps.is_built())