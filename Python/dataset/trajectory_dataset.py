import numpy as np
import torch

class TrajectoryDataset(torch.utils.data.Dataset):

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

    def __init__(self, min_subseq_length : int, max_subseq_length : int, filepath : str, needs_conversion : bool = False):
        self.min_subseq_length = min_subseq_length
        self.max_subseq_length = max_subseq_length
        # assume that filepath is \\dataset.pt
        self.filepath = filepath
        self.expert_trajectories = []
        if needs_conversion:
            self._load_convert_save()
        else:
            self._load()
    
    def _load(self):
        taus = torch.load(self.filepath)
        # leave out seqs that are shorter or longer than provided min/max_subseq_length
        self.expert_trajectories = [t for t in taus if self.min_subseq_length <= len(t)//3 and len(t)//3 <= self.max_subseq_length]

    """
    Take a stored expert trajectory, convert it to the desired format
    and cut into all possible subsequences between (including) min/max_subseq_length
    then save again to save on computation!
    """
    def _load_convert_save(self):
        # load raw trajectories form filepath
        raw_trajectories = torch.load(self.filepath)
        
        # iterate over all trajectories and convert to subsequences
        for tau in raw_trajectories:
            conv_tau = self._convert_tau(tau)
            cut_taus = self._cut_tau(conv_tau)
        
        # add to expert trajectories
        self.expert_trajectories += cut_taus

        # sort by length?
        sorted_taus = sorted(self.expert_trajectories, key = lambda x: len(x))

        # save as .pt file at given save filepath
        torch.save(sorted_taus, "".join(self.filepath.split('.')[:-1]+["_converted.pt"]))

    def _convert_tau(self, tau):
        # check how many actions did happen within this trajectory
        nr_steps = len(tau) // 3
        # new trajectory as empty list
        full_tau = []
        # aggregator for return-to-go
        returntogo = 0

        # iterate from last!! state to first state, so it is easier to compute the return-to-go
        # each packet has (state, action, reward)
        for i in range(nr_steps-1, -1, -1):
            shifted_idx = 3*i
            state = tau[shifted_idx]
            action = tau[shifted_idx + 1]
            reward = tau[shifted_idx + 2]

            # split state into grid and blocks
            grid_flat, b1_flat, b2_flat, b3_flat = state[:100], state[100:125], state[125:150], state[150:]

            # create state as list of these flattend grids
            state_ = [grid_flat, b1_flat, b2_flat, b3_flat]

            # add to beginning of the new tau
            full_tau = [state_, action, returntogo] + full_tau

            # compute return to go for the next iteration
            returntogo += reward

        return full_tau

    def _cut_tau(self, tau : list) -> [list]:
        cut_taus = []
        # iterate over all possible subsequence lengts
        # check if current tau has the max and length!
        
        # if tau does not have the minimum length, return empty list 
        if len(tau) // 3 < self.min_subseq_length:
            return []
        
        max_subseq_length = min(len(tau) // 3, self.max_subseq_length)

        for length in range(self.min_subseq_length, max_subseq_length+1, 1):
            # iterate from end to beginning with window of size length
            # change RTG
            cut_taus += self._subseq_tau(tau, length)
        
        return cut_taus

    def _subseq_tau(self, tau : list, seq_len : int) -> [list]:
        # create all subsequences of the given length for this tau
        sub_seq = []
        nr_steps = len(tau) // 3
        # sequence starts at 3*(nr_steps-subseq_len)
        # length is 3*subseq_len
        for i in range(nr_steps-seq_len, -1, -1):
            shifted_idx_low = 3*i
            shifted_idx_high = shifted_idx_low + 3* seq_len
            seq = tau[shifted_idx_low:shifted_idx_high]
            # change all RTG!
            rtg_diff = seq[-1]
            for j in range(seq_len-1, -1, -1):
                shifted_idx = 3*j+2
                seq[shifted_idx] -= rtg_diff
            
            sub_seq += [seq]
        
        return sub_seq

    def __len__(self) -> int:
        return len(self.expert_trajectories)
    
    
    def __getitem__(self, idx) -> list:
        return self.expert_trajectories[idx]