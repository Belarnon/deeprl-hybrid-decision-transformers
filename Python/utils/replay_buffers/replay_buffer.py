from abc import ABC, abstractmethod
from tensordict import TensorDict
from typing import List, Iterable

_ABSTRACT_CLASS_ERROR_MSG = "ReplayBuffer is an abstract class and should not be instantiated directly."

class AbstractReplayBuffer(ABC):
    """
    Abstract class for replay buffers to store
    and sample sequences of experiences.
    """

    @abstractmethod
    def add(self, sequence : TensorDict) -> None:
        """
        Add a sequence of experiences to the replay buffer.

        sequence: sequence of experiences to add
        """
        raise NotImplementedError(_ABSTRACT_CLASS_ERROR_MSG)

    @abstractmethod
    def extend(self, sequence : TensorDict) -> None:
        """
        Extend the replay buffer with a sequence of experiences.

        sequence: sequence of experiences to add
        """
        raise NotImplementedError(_ABSTRACT_CLASS_ERROR_MSG)

    @abstractmethod
    def sample(self, batch_size : int) -> List[TensorDict]:
        """
        Sample a batch of sequences from the replay buffer.

        batch_size: number of sequences to sample
        """
        raise NotImplementedError(_ABSTRACT_CLASS_ERROR_MSG)
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of sequences in the replay buffer.
        """
        raise NotImplementedError(_ABSTRACT_CLASS_ERROR_MSG)
    
    @abstractmethod
    def __iter__(self) -> Iterable[List[TensorDict]]:
        """
        Return an iterable object to iterate over batches of sequences in the replay buffer.
        """
        raise NotImplementedError(_ABSTRACT_CLASS_ERROR_MSG)
    
    @abstractmethod
    def __next__(self) -> List[TensorDict]:
        """
        Return the next batch of sequences in the replay buffer.
        """
        raise NotImplementedError(_ABSTRACT_CLASS_ERROR_MSG)
    
    @abstractmethod
    def __getitem__(self, idx) -> TensorDict:
        """
        Return the sequence at the given index.
        """
        raise NotImplementedError(_ABSTRACT_CLASS_ERROR_MSG)
        
        