import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from utils.transformer_block import TransformerBlock

class DecisionTransformer(nn.Module):

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=128, # aka embedding_dim
            max_length=None, # the number of timesteps to consider in the past
            max_episode_length=4096, # the maximum number of timesteps in an episode/trajectory
            action_tanh=True
            ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        
        self.transformer = TransformerBlock(
            self.hidden_dim,
            heads=4,
            n_mlp=2
        )

        self.embed_timestep = nn.Embedding(max_episode_length, self.hidden_dim)
        self.embed_action = nn.Linear(self.action_dim, self.hidden_dim)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_dim)
        self.embed_return = nn.Linear(1, self.hidden_dim)

        self.embed_ln = nn.LayerNorm(self.hidden_dim)

        # TODO change linear decoding for action for cross-entropy loss? edit: is this still relevant?

        self.predict_return = nn.Linear(self.hidden_dim, 1)
        self.predict_action = nn.Sequential(*(
            [nn.Linear(self.hidden_dim, self.action_dim)] # predicted actions are always one-hot encoded
            + ([nn.Tanh()] if action_tanh else [])
        ))
        self.predict_state = nn.Linear(hidden_dim, self.state_dim)


    def forward(self, states: torch.tensor, actions: torch.tensor, returns_to_go: torch.tensor, timesteps: torch.tensor, attention_mask: torch.tensor=None):
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
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(states.device)

        # embed each modality with a different head
        
        wa söllemer als timesteps übergeh? sinds ez nur die vode einzelne steps oder vode max_episode_length?
        schaltet sie s'nögste mol ii bi 10vor10 und findends use
        als spezialgast de jumbo schreiner wo es schnitzel isst.

        timestep_embeddings = self.embed_timestep(timesteps) # shape (batch_size, seq_length, hidden_dim)
        action_embeddings = self.embed_action(actions)
        state_embeddings = self.embed_state(states)
        return_embeddings = self.embed_return(returns_to_go)

        # time embeddings are treated similar to positional embeddings
        action_embeddings += timestep_embeddings
        state_embeddings += timestep_embeddings
        return_embeddings += timestep_embeddings

        # stack & interleave embeddings: [r_0, s_0, a_0, r_1, s_1, a_1, ...] (for each element in the batch)
        embeddings = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_dim) # shape (batch_size, 3*seq_length, hidden_dim)
        embeddings = self.embed_ln(embeddings)

        # do the same for the attention mask
        attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # TODO: correctly use transformer (or use different transformer altogether, aka GPT2)
        x = self.transformer(embeddings, attention_mask)

        # reshape x to reverse the stacking & interleaving
        x = x.reshape(batch_size, seq_length, 3, self.hidden_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state
        # the above comments were written by the original authors, but we are not sure if they are correct

        return return_preds, state_preds, action_preds
