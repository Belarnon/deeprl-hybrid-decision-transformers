import torch
import torch.nn as nn

from utils.transformer_block import TransformerBlock

class DecisionTransformer(nn.Module):

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=128, # aka embedding_dim
            max_length=None, # the number of timesteps to consider in the past
            max_episode_length=4096, # the maximum number of timesteps in an episode/trajectory
            action_tanh=True,
            **kwargs
            ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.hidden_dim = hidden_dim

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


    def forward(self, states: torch.Tensor, actions: torch.Tensor, returns_to_go: torch.Tensor, timesteps: torch.Tensor, attention_mask: torch.Tensor=None):
        """
        Perform a forward pass through the transformer.

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

        attention_mask = attention_mask.bool()
        if len(attention_mask.shape) == 2: # nn.Transformer expects either a (l,l) or (b,l,l) mask, but can't handle (b,l) masks
            attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.shape[1], 1) # broadcast attention vector to matrix (copy row-wise)

        x = self.transformer(embeddings) # TODO: currently results in nan values if attention_mask is passed

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

        states: shape (batch_size=1, seq_length, state_dim)
        actions: shape (batch_size=1, seq_length, action_dim)
        returns_to_go: shape (batch_size=1, seq_length, 1)
        timesteps: shape (batch_size=1, seq_length)
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
                (torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1]))
                ).to(dtype=torch.long, device=states.device).reshape(1, -1)
        else:
            attention_mask = None


        # Perform a forward pass through the transformer.
        _, _, action_preds = self.forward(states, actions, returns_to_go, timesteps, attention_mask)

        # Return last action prediction
        return action_preds[0, -1] # return last action prediction
