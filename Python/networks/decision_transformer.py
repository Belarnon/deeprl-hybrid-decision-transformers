import numpy as np
import torch
import torch.nn as nn

from ..dataset.trajectory_dataset import TrajectoryDataset

class DecisionTransformer(nn.Module):

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=128, # aka embedding_dim
            max_length=None,
            max_episode_length=None,
            action_tanh=True
            ):
        super().__init__(state_dim, action_dim, max_length)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.hidden_dim = hidden_dim

        self.embed_timestep = nn.Embedding(max_episode_length, self.hidden_dim)
        self.embed_action = nn.Embedding(self.action_dim, self.hidden_dim)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_dim)
        self.embed_return = nn.Linear(1, self.hidden_dim)

        self.predict_return = nn.Linear(self.hidden_dim, 1)
        self.predict_action = nn.Sequential(*(
            [nn.Linear(self.hidden_dim, self.action_dim)]
            + ([nn.Tanh()] if action_tanh else [])
        ))
        self.predict_state = nn.Linear(hidden_dim, self.state_dim)
    
    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        """
        states: shape (batch_size, seq_length, state_dim)
        actions: shape (batch_size, seq_length, action_dim)
        returns_to_go: shape (batch_size, seq_length, 1)
        timesteps: shape (batch_size, seq_length)
        attention_mask: shape (batch_size, seq_length)
        """

        batch_size, seq_length = states.shape[:2]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        timestep_embeddings = self.embed_timestep(timesteps) # shape (batch_size, seq_length, hidden_dim)
        action_embeddings = self.embed_action(actions)
        state_embeddings = self.embed_state(states)
        return_embeddings = self.embed_return(returns_to_go)

        # time embeddings are treated similar to positional embeddings
        action_embeddings += timestep_embeddings
        state_embeddings += timestep_embeddings
        return_embeddings += timestep_embeddings

        # TODO: Do predictions *correctly*, the following is just a placeholder

        # concatenate embeddings
        embeddings = torch.cat(
            (state_embeddings, action_embeddings, return_embeddings),
            dim=1
        ) # shape (batch_size, 3*seq_length, hidden_dim)

        # get predictions
        next_state = self.predict_state(embeddings) # shape (batch_size, seq_length, state_dim)
        next_action = self.predict_action(embeddings) # shape (batch_size, seq_length, action_dim)
        next_return = self.predict_return(embeddings) # shape (batch_size, seq_length, 1)

        return next_state, next_action, next_return

