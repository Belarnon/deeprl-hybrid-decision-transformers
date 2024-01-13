import numpy as np
import torch
import json
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):

    """
    This class loads the list of trajectories stored from the `TrajectoryDatasetCreator` class in trajectory_gym.py
    and turn them into a usable dataset

    ds_creator: [tau_0, tau_1,...,tau_m]
    tau_i = [s_0, a_0, r_0, .. , s_T, a_T, r_T] where s_{t+1} = S(s_t, a_t), r_t = R(s_t, a_t)
    s_i = [flat_grid, flat_blck_1, flat_block_2, flat_block_3]
    r_i = float

    IMPORTANT: grid and blocks are flattend along rows, from top to bottom (Z direction)
    
    TrajectoryDataset: [tau'_0, tau'_1,...,tau'_m]
    tau'_i = [s'_0, a_0, R_0, .. , s'_T, a_T, R_T]
    s'_i = [flat_grid, flat_block_1, flat_block_2, flat_block_3]
    R_i = sum_{t'=t}^T r_{t'}

    """

    def __init__(self, min_subseq_length : int, max_subseq_length : int, stride : int, filepath : str):
        self.min_subseq_length = min_subseq_length
        # max is set later!
        #self.max_subseq_length = max_subseq_length
        self.stride = stride
        self.filepath = filepath

        # load json, extract list of trajectories
        # check if given file has .json ending
        file_ending = self.filepath.split('.')[-1]
        if file_ending == 'json':
            with open(file=self.filepath) as file:
                taus_json = json.load(file)['trajectories']
        else:
            raise NotImplementedError(f"File ending is unknown! {file_ending}")
    
        # get list of trajectories, where every trajectory is a list of transitions
        # remove empty trajectories
        self.taus = [t['transitions'] for t in taus_json if len(t['transitions']) != 0]
        # get lengths of trajectories
        self.taus_lengths = [len(t) for t in self.taus]
        # adjust max_subseq_len by maximum possible sequence
        self.max_subseq_length = min(max_subseq_length, max(self.taus_lengths))
        
        # compute number of all subsequences and fill LookUp-Table for number
        # of subsequences for each sequence and length
        self.LUT = self._compute_subseq_LUT()
        self.total_nr_subseq =  int(self.LUT.sum())

    def _compute_subseq_LUT(self) -> np.ndarray:
        # compute number of different subsequence lengths
        nr_subseq_lengths = self.max_subseq_length - self.min_subseq_length + 1
        # create table with dim = (nr_taus, nr_subseq_lengths)
        lookuptable = np.zeros((len(self.taus), nr_subseq_lengths), dtype=int)
        
        # iterate over all trajectory lengths
        for i, tau_len in enumerate(self.taus_lengths):
            # iterate over all subseq lengths
            for j, seq_len in enumerate(list(range(self.min_subseq_length, self.max_subseq_length+1))):
                overlap = seq_len - self.stride
                lookuptable[i,j] = (tau_len - overlap) // self.stride

        return lookuptable


    def __len__(self) -> int:
        return self.total_nr_subseq
    

    def __getitem__(self, idx: int) -> list:

        # from index compute tau:
        # - which trajectory
        # - which sequence length
        # - 
       
        # check via look up table which sequence and which subsequence length to take
        inter_idx = idx
        tau_idx, seqlen_idx = -1,-1
        is_finished = False
        # nested for-loop iteration over lookuptable row-wise to find traj and seq len
        for i in range(len(self.taus)):
            if is_finished: break
            for j in range(self.max_subseq_length - self.min_subseq_length + 1):
                # get number of trajectories for this tau and seqlen
                # if inter_idx is bigger than it, subtract number of sequences and go on
                # else the inter_idx is the index of th subseq for the tau and subseq len
                curr_nr_subseq = self.LUT[i, j]
                if inter_idx >= curr_nr_subseq:
                    inter_idx -= curr_nr_subseq
                else:
                    tau_idx, seqlen_idx = i,j
                    is_finished = True
                    break

        # get computed tau and seq length
        # seq_len is min_seq_len + index
        tau, seq_len = self.taus[tau_idx], self.min_subseq_length + seqlen_idx

        # use inter_idx to get starting index using the stride
        # also compute end index
        start_idx = inter_idx * self.stride
        end_idx = start_idx + seq_len
        # get subsequence from tau
        subseq = tau[start_idx:end_idx]

        # convert tau from json format to separate arrays
        state_dim = len(tau[0]['observation'])
        action_dim = len(tau[0]['action']['discreteActions'])
        reward_dim = 1 if type(tau[0]['reward']) is float else len(tau[0]['reward'])

        states = np.zeros((seq_len, state_dim))
        actions = np.zeros((seq_len, action_dim))
        rewards = np.zeros((seq_len, reward_dim))

        # iterate BACKWARDS over subseq and fill arrays
        target_rtg = subseq[-1]['reward']
        # return to go: take last reward and subtract from every reward to get return-to-go
        for i in range(seq_len-1, -1, -1):
            states[i] = subseq[i]['observation']
            actions[i] = subseq[i]['action']['discreteActions']
            rewards[i] = subseq[i]['reward'] - target_rtg

        # timesteps
        timesteps = np.linspace(1, seq_len, seq_len)

        #pad from left with zeros to max_subseq_length
        pad_steps = self.max_subseq_length - seq_len

        states = np.concatenate([np.zeros((pad_steps, state_dim)), states])
        actions = np.concatenate([np.zeros((pad_steps, action_dim)), actions])
        rewards = np.concatenate([np.zeros((pad_steps, reward_dim)), rewards])
        timesteps = np.concatenate([np.zeros(pad_steps), timesteps])
        attention_mask = np.concatenate([np.zeros(pad_steps), np.ones(seq_len)])

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        timesteps = torch.from_numpy(timesteps).int()
        attention_mask = torch.from_numpy(attention_mask)

        return [states, actions, rewards, timesteps, attention_mask]