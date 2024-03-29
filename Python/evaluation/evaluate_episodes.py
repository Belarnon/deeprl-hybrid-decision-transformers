"""
This code is adapted from the following repositories:

https://github.com/kzl/decision-transformer

Modified to include additional functionality.
"""

import numpy as np
import torch

from utils.training_utils import decode_actions

def get_action_hugging(model: torch.nn.Module, states: torch.Tensor, actions: torch.Tensor, returns_to_go: torch.Tensor, timesteps: torch.Tensor):
        """
        Get the action prediction for a given state, action, return, and timestep. Used for evaluation.
        If batch size > 1, the sequences are simply concatenated along the batch dimension.

        Args:
            model (torch.nn.Module): The pretrained huggingface model
            states (torch.Tensor): shape (batch_size, seq_length, state_dim)
            actions (torch.Tensor): shape (batch_size, seq_length, action_dim)
            returns_to_go (torch.Tensor): shape (batch_size, seq_length, 1)
            timesteps (torch.Tensor): shape (batch_size, seq_length)

        Returns:
            action_pred (torch.Tensor): Predicted action, still one-hot encoded, shape (action_dim,)
        """

        # Get model parameters
        state_dim = model.config.state_dim
        action_dim = model.config.act_dim
        max_length = model.config.max_length


        # Reshape inputs to fit the model in case batch_size != 1
        # In that case, the sequences are simply concatenated along the batch dimension
        states = states.reshape(1, -1, state_dim)
        actions = actions.reshape(1, -1, action_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # Trim / add padding to the sequences to fit the model
        if max_length is not None:
            # Trim sequences to max_length
            states = states[:, -max_length:]
            actions = actions[:, -max_length:]
            returns_to_go = returns_to_go[:, -max_length:]
            timesteps = timesteps[:, -max_length:]

            # TODO transformer only needs last K observations
            # instead of COPYING the complete sequence to the transformer, cut it here?
             
            # Add left padding to the sequences
            states = torch.cat(
                (torch.zeros(states.shape[0], max_length - states.shape[1], state_dim, device=states.device), states),
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                (torch.zeros(actions.shape[0], max_length - actions.shape[1], action_dim, device=actions.device), actions),
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                (torch.zeros(returns_to_go.shape[0], max_length - returns_to_go.shape[1], 1, device=returns_to_go.device), returns_to_go),
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                (torch.zeros(timesteps.shape[0], max_length - timesteps.shape[1], device=timesteps.device), timesteps),
                dim=1).to(dtype=torch.float) # Maybe we need to change this to int in the future :thinking:
            attention_mask = torch.cat(
                (torch.zeros(max_length - states.shape[1]), torch.ones(states.shape[1]))
                ).to(dtype=torch.float, device=states.device).reshape(1, -1)
        else:
            attention_mask = None


        # Perform a forward pass through the transformer.
        _, _, action_preds = model.forward(states, actions, returns_to_go, timesteps, attention_mask) # TODO: Doesn't work yet for some reason, dimensions don't match

        # Return last action prediction
        return action_preds[0, -1] # return last action prediction


def evaluate_episode_rtg(
        env,
        state_dim,
        action_dim,
        model,
        device,
        target_return,
        nr_episodes=10,
        action_space=(3, 10, 10),
        target_return_scaling=1,
        use_huggingface=False
    ):

    model.eval()
    model.to(device=device)

    # evaluation loop consists of multiple games
    episodes_returns = torch.zeros(nr_episodes).to(device, dtype=torch.float32)
    episodes_lengths = torch.zeros(nr_episodes).to(device, dtype=torch.float32)

    # for online training we need to keep track of the trajectory
    # for now only the last one is saved, maybe save them all or the best in the future
    trajectory = {}

    for e in range(nr_episodes):

        state = env.reset()
    
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, action_dim), device=device, dtype=torch.float32)
        actions_decoded = torch.zeros((0, len(action_space)), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        returns_to_go = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        episode_return, episode_length, t = 0, 0, 0

        while True:
            # add padding
            actions = torch.cat([actions, torch.zeros((1, action_dim), device=device)], dim=0)
            actions_decoded = torch.cat([actions_decoded, torch.zeros((1, len(action_space)), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            if use_huggingface:
                action = get_action_hugging(
                    model,
                    states.to(dtype=torch.float32),
                    actions.to(dtype=torch.float32),
                    returns_to_go.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                    )
            else:
                action = model.get_action(
                    states.to(dtype=torch.float32),
                    actions.to(dtype=torch.float32),
                    returns_to_go.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
            actions[-1] = action
            action = decode_actions(action.reshape(1, 1, -1), action_space)
            actions_decoded[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)
            reward = torch.tensor(reward)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = returns_to_go[0,-1] - (reward/target_return_scaling)
            returns_to_go = torch.cat(
                [returns_to_go, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1
            t += 1

            if done:
                episodes_returns[e] = episode_return
                episodes_lengths[e] = episode_length
                break
        
        # Unity sends a final observation after the episode is done
        # we need to remove this observation
        states = states[:-1]

        # save trajectory (only the last one is saved)
        trajectory.update(
            {
                "observations": states, # shape (seq_len, state_dim)
                "actions": actions_decoded, # shape (seq_len, len(action_space))
                "rewards": rewards, # shape (seq_len,)
            }
        )

    return episodes_returns, episodes_lengths, trajectory
