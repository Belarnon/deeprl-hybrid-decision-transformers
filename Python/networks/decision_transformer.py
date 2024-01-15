"""
This code is adapted from the following repositories:

https://github.com/kzl/decision-transformer
https://github.com/facebookresearch/online-dt

Modified to include additional functionality.
"""

import torch
import torch.nn as nn

from xformers.factory import xFormerConfig, xFormer

from modules.observation_encoder import ObservationEncoder, GridEncoder, BlockEncoder, ResidualMLP

class DecisionTransformer(nn.Module):

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
            use_xformers=False, # whether to use xFormers instead of PyTorch's transformer
            ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.max_episode_length = max_episode_length
        self.hidden_dim = hidden_dim

        self.grid_size = grid_size
        self.block_size = block_size
        self.grid_size_squared = self.grid_size*self.grid_size
        self.block_size_squared = self.block_size*self.block_size

        self.use_xformers = use_xformers

        if self.use_xformers:
            # xFormers transformer encoder
            config = xFormerConfig([{
                "block_type": "encoder",
                "num_layers": 3,
                "dim_model": self.hidden_dim,
                "residual_norm_style": "pre",  # Optional, pre/post
                "multi_head_config": {
                    "num_heads": 1,
                    "residual_dropout": 0.1,
                    "attention": {
                        "name": "linformer",  # whatever attention mechanism
                        "dropout": 0.1,
                        "seq_len": 3 * self.max_length, # times 3 because on the attention level, we will have interleaved rtg, state, and action
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": 0.1,
                    "activation": "gelu",
                    "hidden_layer_multiplier": 4,
                },
            }])
            self.xformer = xFormer.from_config(config)
        else:
            # PyTorch's transformer encoder
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=1,
                    dim_feedforward=4*self.hidden_dim,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True
                ),
                num_layers=3
            )

        # CNNs for encoding grid and block observations
        self.grid_cnn = GridEncoder()
        self.block_cnn = BlockEncoder()
        self.res_mlp = ResidualMLP(self.hidden_dim, self.hidden_dim)
 
        # Define linear layers
        self.embed_timestep = nn.Embedding(self.max_episode_length, self.hidden_dim)
        self.embed_action = nn.Linear(self.action_dim, self.hidden_dim)
        if fancy_look_embedding:
            self.embed_state = ObservationEncoder(self.grid_cnn, self.block_cnn, self.res_mlp, self.grid_size, self.block_size)
        else:
            self.embed_state = nn.Linear(self.state_dim, self.hidden_dim)
        self.embed_return = nn.Linear(1, self.hidden_dim)

        self.embed_ln = nn.LayerNorm(self.hidden_dim)

        self.predict_return = nn.Linear(self.hidden_dim, 1)
        self.predict_action = nn.Sequential(*(
            [nn.Linear(self.hidden_dim, self.action_dim)] # predicted actions are always one-hot encoded
            + ([nn.Tanh()] if action_tanh else [])
        ))
        self.predict_state = nn.Linear(hidden_dim, self.state_dim)


    def forward(self, states: torch.Tensor, actions: torch.Tensor, returns_to_go: torch.Tensor, timesteps: torch.Tensor, padding_masks: torch.Tensor=None):
        """
        Perform a forward pass through the transformer.

        Args:
            states (torch.Tensor): shape (batch_size, seq_length, state_dim)
            actions (torch.Tensor): shape (batch_size, seq_length, action_dim)
            returns_to_go (torch.Tensor): shape (batch_size, seq_length, 1)
            timesteps (torch.Tensor): shape (batch_size, seq_length)
            padding_masks (torch.Tensor): shape (batch_size, seq_length)

        Returns:
            return_preds (torch.Tensor): shape (batch_size, seq_length, 1)
            state_preds (torch.Tensor): shape (batch_size, seq_length, state_dim)
            action_preds (torch.Tensor): shape (batch_size, seq_length, action_dim)
        """

        batch_size, seq_length = states.shape[:2]

        # Padding masks are used to mask out padding tokens in the transformer
        if padding_masks is None:
            # attention mask for nn.TransformerEncoder: 0 if can be attended to, 1 if not
            padding_masks = torch.zeros((batch_size, seq_length), dtype=torch.long).to(states.device)
        
        # Attention mask for ignoring future states. Upper triangular matrix with above-diagonal elements set to 1
        attention_mask = nn.Transformer.generate_square_subsequent_mask(3 * seq_length).to(states.device)

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

        # do the same for the padding mask
        padding_masks = torch.stack(
            (padding_masks, padding_masks, padding_masks), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        attention_mask = attention_mask.bool()
        padding_masks = padding_masks.bool()
        if self.use_xformers:
            padding_masks = ~padding_masks # xFormers uses 1 for attention and 0 for no attention
            x = self.xformer(embeddings, encoder_input_mask=padding_masks)
        else:
            # since every sequence has its own attention mask vector we need to give it the transformer as mask
            x = self.transformer(embeddings, is_causal=True, mask=attention_mask, src_key_padding_mask=padding_masks)

        # reshape x to reverse the stacking & interleaving
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state
        # "given state and action"
        # only gives action
        # ???
        # these comments are confusing, but they are from the original code, so...

        return return_preds, state_preds, action_preds
    
    
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
            action_pred (torch.Tensor): Predicted action, still one-hot encoded, shape (action_dim,)
        """

        # Reshape inputs to fit the model in case batch_size != 1
        # In that case, the sequences are simply concatenated along the batch dimension
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # Trim / add padding to the sequences to fit the model
        if self.max_length is not None:
            # Trim sequences to max_length
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # Add left padding to the sequences
            states = torch.cat(
                (torch.zeros(states.shape[0], self.max_length - states.shape[1], self.state_dim, device=states.device), states),
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                (torch.zeros(actions.shape[0], self.max_length - actions.shape[1], self.action_dim, device=actions.device), actions),
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                (torch.zeros(returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1, device=returns_to_go.device), returns_to_go),
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                (torch.zeros(timesteps.shape[0], self.max_length - timesteps.shape[1], device=timesteps.device), timesteps),
                dim=1).to(dtype=torch.long) # Maybe we need to change this to int in the future :thinking:
            attention_mask = torch.cat(
                (torch.ones(self.max_length - states.shape[1]), torch.zeros(states.shape[1]))
                ).to(dtype=torch.long, device=states.device).reshape(1, -1)
        else:
            attention_mask = None


        # Perform a forward pass through the transformer.
        _, _, action_preds = self.forward(states, actions, returns_to_go, timesteps, attention_mask)

        # Return last action prediction
        # Since we might have overwritten the action prediction layer, we need to check for that
        if isinstance(action_preds, torch.distributions.Distribution):
            return action_preds.sample()[0, -1] # return last action prediction
        return action_preds[0, -1] # return last action prediction

if __name__ == "__main__":
    # Test the model
    batches = 2
    seq_length = 10
    state_dim = 175
    action_dim = 23
    model = DecisionTransformer(state_dim, action_dim, max_length=10)
    states = torch.rand(batches, seq_length, state_dim)
    actions = torch.rand(batches, seq_length, action_dim)
    returns_to_go = torch.rand(batches, seq_length, 1)
    timesteps = torch.arange(seq_length).repeat(batches, 1)
    return_preds, state_preds, action_preds = model(states, actions, returns_to_go, timesteps)
    print(return_preds.shape)
    print(state_preds.shape)
    print(action_preds.shape)
    
    # Make sure we have no nans
    assert not torch.isnan(return_preds).any()
    assert not torch.isnan(state_preds).any()
    assert not torch.isnan(action_preds).any()
