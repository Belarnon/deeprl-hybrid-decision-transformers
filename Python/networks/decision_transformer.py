import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from ..utils.transformer_block import TransformerBlock

class DecisionTransformer(nn.Module):

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=128, # aka embedding_dim
            max_length=None, # the number of timesteps to consider in the past
            max_episode_length=4096, # the maximum number of timesteps in an episode/trajectory
            action_tanh=True,
            action_space=(3,10,10),
            encode_actions=True # whether to one-hot encode actions
            ):
        super().__init__(state_dim, action_dim, max_length)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_encoded_dim = sum(action_space)
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.encode_actions = encode_actions
        
        self.transformer = TransformerBlock(
            self.hidden_dim,
            n_heads=4,
            n_layers=2,
            attention_mask=True
        )

        self.embed_timestep = nn.Embedding(max_episode_length, self.hidden_dim)
        self.embed_action = nn.Linear((self.action_encoded_dim if encode_actions else self.action_dim), self.hidden_dim) # use appropriate input dim depending on `encode_actions`
        self.embed_state = nn.Linear(self.state_dim, self.hidden_dim)
        self.embed_return = nn.Linear(1, self.hidden_dim)

        self.embed_ln = nn.LayerNorm(self.hidden_dim)

        # TODO change linear decoding for action for cross-entropy loss?

        self.predict_return = nn.Linear(self.hidden_dim, 1)
        self.predict_action = nn.Sequential(*(
            [nn.Linear(self.hidden_dim, self.action_encoded_dim)] # predicted actions are always one-hot encoded
            + ([nn.Tanh()] if action_tanh else [])
        ))
        self.predict_state = nn.Linear(hidden_dim, self.state_dim)


    def forward(self, states: torch.tensor, actions: torch.tensor, returns_to_go: torch.tensor, timesteps: torch.tensor, attention_mask=None):
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

        if self.encode_actions:
            # one-hot encode actions
            max_classes = max(self.action_space) # the maximum number of classes needed for one-hot encoding
            one_hot_actions = one_hot(actions, max_classes).float() # one hot encoding for max_classes, which might be larger than the actual action space
            flat_actions = one_hot_actions.reshape(batch_size, seq_length, -1) # flatten one-hot encoding
            splice_actions_list = []
            for i in range(len(self.action_space)):
                start = i*max_classes
                end = start + self.action_space[i]
                splice_actions_list.append(flat_actions[:,:,start:end]) # only keep the one-hot encoding for the actual action space
            actions = torch.cat(splice_actions_list, dim=-1)


        # embed each modality with a different head
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
        
        # apply transformer
        embeddings = self.transformer(embeddings, attention_mask)

        # get predictions
        next_state = self.predict_state(embeddings) # shape (batch_size, seq_length, state_dim)
        next_action = self.predict_action(embeddings) # shape (batch_size, seq_length, action_dim)
        next_return = self.predict_return(embeddings) # shape (batch_size, seq_length, 1)

        return next_state, next_action, next_return


    def decode_actions(self, actions: torch.tensor):
        """
        actions: shape (batch_size, seq_length, action_encoded_dim), where actions are one-hot encoded
        """

        target_dim = len(self.action_space)

        # Splice the batch
        decoded_batch_list = []
        for i in range(target_dim):
            start_index = sum(self.action_space[:i])
            end_index = start_index + self.action_space[i]
            decoded_batch_list.append(actions[:,:,start_index:end_index].argmax(-1))

        return torch.stack(decoded_batch_list, -1)
