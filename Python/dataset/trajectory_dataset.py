import numpy as np
import torch
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

    def __init__(self, min_subseq_length : int, max_subseq_length : int, filepath : str, needs_conversion : bool = False, device = None):
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
        tau_ts = torch.load(self.filepath)
        # leave out seqs that are shorter or longer than provided min/max_subseq_length
        self.expert_trajectories = [t for t in tau_ts if self.min_subseq_length <= len(t[0])//3 and len(t[0])//3 <= self.max_subseq_length]

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
            tau_t = self._convert_tau(tau)
            cut_tau_ts = self._cut_tau(tau_t)
        
            # add to expert trajectories
            self.expert_trajectories += cut_tau_ts

        # sort by length?
        #sorted_taus = sorted(self.expert_trajectories, key = lambda x: len(x))

        # save as .pt file at given save filepath
        torch.save(self.expert_trajectories, "".join(self.filepath.split('.')[:-1]+["_converted.pt"]))

    
    def _convert_tau(self, tau):
        """
        Computes return-to-go for trajectory and adds timesteps
        """

        # check how many actions did happen within this trajectory
        nr_steps = len(tau) // 3
        # new trajectory as empty list
        tau_t = []
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
            #grid_flat, b1_flat, b2_flat, b3_flat = state[:100], state[100:125], state[125:150], state[150:]

            # create state as list of these flattend grids
            #state_ = [grid_flat, b1_flat, b2_flat, b3_flat]

            # add to beginning of the new tau
            tau_t = [state, action, returntogo] + tau_t

            # compute return to go for the next iteration
            returntogo += reward

        # add timesteps array to tau!
        tau_t = [tau_t, np.linspace(0, nr_steps-1, nr_steps)]

        return tau_t

    def _cut_tau(self, tau_t : list) -> [list]:
        cut_tau_ts = []
        # iterate over all possible subsequence lengts
        # check if current tau has the max and length!
        
        # if tau does not have the minimum length, return empty list 
        if len(tau_t[1]) // 3 < self.min_subseq_length:
            return []
        
        max_subseq_length = min(len(tau_t[0]) // 3, self.max_subseq_length)

        for length in range(self.min_subseq_length, max_subseq_length+1, 1):
            # iterate from end to beginning with window of size length
            # change RTG
            cut_tau_ts += self._subseq_tau(tau_t, length)
        
        return cut_tau_ts

    def _subseq_tau(self, tau_t : list, seq_len : int) -> [list]:
        # separate tau from timesteps
        tau, timesteps = tau_t
        # create all subsequences of the given length for this tau and timesteps
        sub_seq = []
        nr_steps = len(tau) // 3
        # sequence starts at 3*(nr_steps-subseq_len)
        # length is 3*subseq_len
        for i in range(nr_steps-seq_len, -1, -1):
            shifted_idx_low = 3*i
            shifted_idx_high = shifted_idx_low + 3* seq_len
            seq = tau[shifted_idx_low:shifted_idx_high]
            times = timesteps[i:i+seq_len]
            # change all RTG!
            rtg_diff = seq[-1]
            for j in range(seq_len-1, -1, -1):
                shifted_idx = 3*j+2
                seq[shifted_idx] -= rtg_diff
            
            sub_seq += [[seq,times]]
        
        return sub_seq

    def __len__(self) -> int:
        return len(self.expert_trajectories)
    
    
    def __getitem__(self, idx: int) -> np.ndarray:
        # get trajectory with timesteps and separate
        tau, timesteps = self.expert_trajectories[idx]

        # take every third element
        # already pad from left with zeros to max_subseq_length
        nr_steps = len(tau) // 3
        pad_steps = self.max_subseq_length - len(tau)//3
        state_dim = len(tau[0])
        action_dim = len(tau[1])

        states = np.concatenate([np.zeros((pad_steps, state_dim)), tau[0::3]])
        actions = np.concatenate([np.zeros((pad_steps, action_dim)), tau[1::3]]) # every third element starting at index 1
        rtg = np.concatenate([np.zeros(pad_steps), tau[2::3]])
        timesteps = np.concatenate([np.zeros(pad_steps), timesteps])
        attention_mask = np.concatenate([np.zeros(pad_steps), np.ones(nr_steps)])

        return [states, actions, rtg, timesteps, attention_mask]