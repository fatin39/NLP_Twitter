import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

print("Is CUDA (GPU support) available in your system?", cuda_available)

# If CUDA is available, it also prints the name of the GPU
if cuda_available:
    print("GPU Name:", torch.cuda.get_device_name(0))

