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

    def __init__(self, path: str):
        self.raw_trajectories = torch.load(path)
        self.expert_trajectories = []
        for tau in self.raw_trajectories:
            self._convert_trajectories(tau)
    
    def _convert_trajectories(self, tau):
        # check how many actions did happen within this trajectory
        nr_steps = len(tau) / 3

        # new trajectory as empty list
        tau_ = []
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
            tau_ = [state_, action, returntogo] + tau_

            # compute return to go for the next iteration
            returntogo += reward
        
        self.expert_trajectories.append(tau_)
    
    def __len__(self) -> int:
        return len(self.expert_trajectories)
    
    
    def __getitem__(self, idx) -> list:
        return self.expert_trajectories[idx]