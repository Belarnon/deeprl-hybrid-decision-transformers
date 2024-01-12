import torch
import numpy as np

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

def encode_actions(action_batch: torch.tensor, action_space=(3,10,10)):
    """
    Encodes actions into one-hot vectors.

    action_batch: batch of action sequences
    action_space: tuple of action space dimensions
    """
    
    encoded_actions = torch.zeros(action_batch.shape[0], action_batch.shape[1], sum(action_space), device=action_batch.device)

    offset = np.insert(np.cumsum(list(action_space)), 0,0)
    for i, sequence in enumerate(action_batch):
        for j, action in enumerate(sequence):
            for k, a in enumerate(action):
                encoded_actions[i, j, offset[k] + int(a)] = 1

    return encoded_actions

def decode_actions(action_batch: torch.tensor, action_space=(3,10,10)):
    """
    Decodes one-hot action vectors into action sequences.

    actions_batch: batch of action sequences
    action_space: tuple of action space dimensions
    """

    decoded_actions = torch.zeros(action_batch.shape[0], action_batch.shape[1], len(action_space), device=action_batch.device)

    limits = np.insert(np.cumsum(list(action_space)), 0,0)
    for i, sequence in enumerate(action_batch):
        for j, action in enumerate(sequence):
            
            act = torch.zeros(len(action_space))
            for k in range(len(action_space)):
                act[k] = action[limits[k]:limits[k+1]].argmax()

            decoded_actions[i, j, :] = act

    return decoded_actions