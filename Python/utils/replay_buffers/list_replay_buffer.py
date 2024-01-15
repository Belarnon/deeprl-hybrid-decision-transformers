from .replay_buffer import AbstractReplayBuffer
from tensordict import TensorDict
from typing import List, Iterable
from random import choices
from math import ceil

class ListReplayBuffer(AbstractReplayBuffer):
    """
    Replay buffer implemented as a simple list of tensor dictionaries 
    representing sequences of experiences.
    """

    def __init__(self, max_seq_num, batch_size, sample_proportional_to_length=True):
        """
        Initialize a ListReplayBuffer object.

        max_seq_num: maximum number of sequences to store
        batch_size: number of sequences to sample in each batch when iterating over the replay buffer
        sample_proportional_to_length: whether to sample sequences with probability proportional to their length
        """
        self.max_seq_num = max_seq_num
        self.batch_size = batch_size
        self.sample_proportional_to_length = sample_proportional_to_length
        self.buffer: List[TensorDict] = []

    def add(self, sequence : TensorDict) -> None:
        """
        Add a sequence of experiences to the replay buffer.

        sequence: sequence of experiences to add
        """
        self.buffer.append(sequence)

        if len(self.buffer) > self.max_seq_num:
            # Remove the oldest sequence
            old = self.buffer.pop(0)
            # Free memory of the oldest sequence
            del old

    def extend(self, sequence : TensorDict) -> None:
        """
        Extend the replay buffer with a sequence of experiences.

        sequence: sequence of experiences to add
        """
        self.buffer.extend(sequence)

        if len(self.buffer) > self.max_seq_num:
            self.buffer = self.buffer[-self.max_seq_num:]

    def sample(self, batch_size : int) -> List[TensorDict]:
        """
        Sample a batch of sequences from the replay buffer.

        batch_size: number of sequences to sample
        """
        return self._sample_batch(batch_size, length_is_prob=self.sample_proportional_to_length)

    def __len__(self) -> int:
        """
        Return the number of sequences in the replay buffer.
        """
        return len(self.buffer)
    
    def __getitem__(self, idx) -> TensorDict:
        """
        Return the sequence at the given index.
        """
        return self.buffer[idx]
    
    def __iter__(self) -> Iterable[List[TensorDict]]:
        """
        Return an iterable object to iterate over batches of sequences in the replay buffer.
        """
        self.batches_to_go = ceil(len(self.buffer) / self.batch_size)
        self.items_left = len(self.buffer)
        return self
    
    def __next__(self) -> List[TensorDict]:
        """
        Return the next batch of sequences in the replay buffer.
        """
        if self.batches_to_go > 0:
            # Compute the batch size for the current batch
            batch_size = min(self.batch_size, self.items_left)
            # Sample a batch of sequences
            batch = self._sample_batch(batch_size, length_is_prob=self.sample_proportional_to_length)
            # Update the number of batches left to iterate over
            self.batches_to_go -= 1
            # Update the number of items left to iterate over
            self.items_left -= batch_size
            # Return the batch
            return batch
        else:
            raise StopIteration
    
    def _sample_batch(self, batch_size : int, length_is_prob: bool = True) -> List[TensorDict]:
        """
        Sample a batch of sequences from the replay buffer.

        batch_size: number of sequences to sample
        length_is_prob: whether to sample sequences with probability proportional to their length
        """

        # Check if the length of the sequences should be sampled with probability proportional to their length
        if length_is_prob:
            # Compute the probability of sampling each sequence
            probs = [seq.shape[0] for seq in self.buffer]

            # Sample sequences
            return choices(self.buffer, weights=probs, k=batch_size)
        
        return choices(self.buffer, k=batch_size)

if __name__ == "__main__":

    # Test the ListReplayBuffer class
    import torch
    
    _SEQ_LEN = 10
    _OBSERVATION_DIM = 20
    _ACTION_DIM = 5
    _REWARD_DIM = 1

    # Create a ListReplayBuffer object
    replay_buffer = ListReplayBuffer(max_seq_num=10, batch_size=4)

    # Add some sequences to the replay buffer
    for i in range(20):
        replay_buffer.add(TensorDict({
            'observations': torch.randn(_SEQ_LEN, _OBSERVATION_DIM),
            'actions': torch.zeros(_SEQ_LEN, _ACTION_DIM),
            'rewards': torch.zeros(_SEQ_LEN, _REWARD_DIM)
        }, batch_size=[_SEQ_LEN]))

    # Iterate over the replay buffer
    for i, batch in enumerate(replay_buffer):
        print(f'Batch {i}: {batch}')