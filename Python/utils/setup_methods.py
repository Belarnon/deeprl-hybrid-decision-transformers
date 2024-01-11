import torch

def find_best_device(use_gpu: bool = False) -> torch.device:
    # Check if we should use the CPU
    if not use_gpu:
        return torch.device('cpu')

    # Check if CUDA is available
    if torch.cuda.is_available():

        # Get all the available GPUs
        gpu_count = torch.cuda.device_count()

        # Check if there are multiple GPUs available
        if gpu_count > 1:
            # Return the first GPU
            device = torch.device('cuda:0')
        else:
            # Return the only GPU
            device = torch.device('cuda')
    else:
        # No CUDA available, return the CPU
        device = torch.device('cpu')

    return device
