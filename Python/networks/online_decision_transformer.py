"""
This code is adapted from the following repositories:

https://github.com/kzl/decision-transformer
https://github.com/facebookresearch/online-dt

Modified to include additional functionality.
"""

import torch
import torch.nn as nn
import numpy as np

from safetensors.torch import load_model

from modules.diag_gaussian_actor import DiagGaussianActor
from networks.decision_transformer import DecisionTransformer

class OnlineDecisionTransformer(nn.Module):
    """
    Online version of the decision transformer. It uses the offline version
    of the decision transformer, but specifies a different action prediction
    using DiagGaussianActor.
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=128, # aka embedding_dim
            max_length=None, # the number of timesteps to consider in the past
            max_episode_length=4096, # the maximum number of timesteps in an episode/trajectory
            action_tanh=True,
            fancy_look_embedding=True, # whether to use state preprocessing using a CNN and more
            grid_size=10, # needed for fancy_look_embedding
            block_size=5, # needed for fancy_look_embedding
            use_xformers=False, # whether to use the xformers library
            pretrained_path=None, # path to pretrained model
            init_temperature=0.1, # initial temperature for the action prediction
            target_entropy=None # target entropy for the action prediction
            ):
        super().__init__()

        self.decision_transformer = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_length=max_length,
            max_episode_length=max_episode_length,
            action_tanh=action_tanh,
            fancy_look_embedding=fancy_look_embedding,
            grid_size=grid_size,
            block_size=block_size,
            use_xformers=use_xformers
        )

        # Load the pretrained model
        if pretrained_path is not None:
            load_model(self.decision_transformer, pretrained_path, strict=False)

        # Overwrite the action prediction
        self.decision_transformer.predict_action = DiagGaussianActor(hidden_dim, action_dim)

        # Initialize the temperature
        self.log_temperature = torch.tensor(np.log(init_temperature))
        self.log_temperature.requires_grad = True
        self.target_entropy = -action_dim if target_entropy is None else target_entropy

    def get_temperature(self):
        return self.log_temperature.exp()
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor, returns_to_go: torch.Tensor, timesteps: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Forward pass through the model. Instead of an action tensor, we now
        return a squashed normal distribution over the action space.

        Args:
            states (torch.Tensor): shape (batch_size, seq_length, state_dim)
            actions (torch.Tensor): shape (batch_size, seq_length, action_dim)
            returns_to_go (torch.Tensor): shape (batch_size, seq_length, 1)
            timesteps (torch.Tensor): shape (batch_size, seq_length)
            attention_mask (torch.Tensor): shape (batch_size, seq_length)

        Returns:
            return_preds (torch.Tensor): shape (batch_size, seq_length, 1)
            state_preds (torch.Tensor): shape (batch_size, seq_length, state_dim)
            action_preds (SquashedNormal): `action_preds.sample()` has
                shape (batch_size, seq_length, action_dim)
        """

        return self.decision_transformer(states, actions, returns_to_go, timesteps, attention_mask)

    def get_action(self, states: torch.Tensor, actions: torch.Tensor, returns_to_go: torch.Tensor, timesteps: torch.Tensor):
        """
        Get the action prediction for a given state, action, return, and timestep. Used for evaluation.
        If batch size > 1, the sequences are simply concatenated along the batch dimension.

        Args:
            states (torch.Tensor): shape (batch_size, seq_length, state_dim)
            actions (torch.Tensor): shape (batch_size, seq_length, action_dim)
            returns_to_go (torch.Tensor): shape (batch_size, seq_length, 1)
            timesteps (torch.Tensor): shape (batch_size, seq_length)

        Returns:
            action_pred (SquashedNormal): Since it is a distribution, we can
                sample a one-hot-encoded action from it using `action_pred.sample()`,
                which has shape (batch_size, seq_length, action_dim).
        """

        return self.decision_transformer.get_action(states, actions, returns_to_go, timesteps)
