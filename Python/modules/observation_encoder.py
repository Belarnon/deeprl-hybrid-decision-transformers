import torch
import torch.nn as nn

class GridEncoder(nn.Module):
    def __init__(self, out_dim: int = 80, batch_norm: bool = True):
        super().__init__()

        # Define CNN layers for grid encoding
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # Assuming grid is a single channel input, shape (batch_size, 8, grid_size, grid_size) (800 per grid)
            nn.BatchNorm2d(8) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # shape (batch_size, 8, grid_size/2, grid_size/2) (200 per grid)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # shape (batch_size, 16, grid_size/2, grid_size/2) (400 per grid)
            nn.BatchNorm2d(16) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # shape (batch_size, 16, grid_size/4, grid_size/4) (64 per grid)
            nn.Flatten(),
            nn.Linear(64, out_dim)
        ) # output shape: (batch_size, out_dim)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Encodes grid observations into a latent vector representation.

        Args:
            grid (torch.Tensor): Grid observations, shape (batch_size * sequence_length, 1, grid_size, grid_size)

        Returns:
            torch.Tensor: Encoded grid observations, shape (batch_size * sequence_length, out_dim)
        """
        return self.grid_cnn(grid)
    
class BlockEncoder(nn.Module):
    def __init__(self, out_dim: int = 16, batch_norm: bool = True):
        super().__init__()

        self.block_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # Assuming each block is a single channel input, shape (batch_size, 8, block_size, block_size) (200 per block)
            nn.BatchNorm2d(8) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # shape (batch_size, 8, block_size/2, block_size/2) (32 per block)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # shape (batch_size, 16, block_size/2, block_size/2) (64 per block)
            nn.BatchNorm2d(16) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # shape (batch_size, 16, block_size/4, block_size/4) (16 per block)
            nn.Flatten(),
            nn.Linear(16, out_dim)
        ) # output shape: (batch_size, 32*(block_size/2)*(block_size/2))

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        """
        Encodes block observations into a latent vector representation.

        Args:
            block (torch.Tensor): Block observations, shape (batch_size * sequence_length, 1, block_size, block_size)

        Returns:
            torch.Tensor: Encoded block observations, shape (batch_size * sequence_length, out_dim)
        """
        return self.block_cnn(block)
    
class ResidualMLP(nn.Module):
    def __init__(self, 
                input_dim: int, 
                output_dim: int, 
                hidden_dim: int = 256, 
                num_hidden_layers: int = 2, 
                layer_norm: bool = True,
                dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm = layer_norm

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            )
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize weights for residual connections
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer[0].weight)
            nn.init.zeros_(layer[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes CNN encoded observations into a latent vector representation.

        Args:
            x (torch.Tensor): Observations, shape (batch_size, sequence_length, input_dim)

        Returns:
            torch.Tensor: Encoded observations, shape (batch_size, sequence_length, output_dim)
        """
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = x + layer(x)

        return self.output_layer(x)
    

class ObservationEncoder(nn.Module):
    def __init__(self, 
                grid_encoder: GridEncoder, 
                block_encoder: BlockEncoder,
                res_mlp: ResidualMLP,
                grid_side_size: int = 10,
                block_side_size: int = 5):
        super().__init__()
        self.grid_encoder = grid_encoder
        self.block_encoder = block_encoder
        self.res_mlp = res_mlp
        self.grid_side_size = grid_side_size
        self.block_side_size = block_side_size
        self.grid_length = grid_side_size * grid_side_size
        self.block_length = block_side_size * block_side_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encodes grid and block observations into a latent vector representation.

        Args:
            state (torch.Tensor): Grid observations, shape (batch_size, sequence_length, state_dim)

        Returns:
            torch.Tensor: Encoded observations, shape (batch_size, sequence_length, out_dim)
        """
        batch_size, sequence_length, _ = state.shape

        # Reshape state into 2D grids and 3x 2D blocks
        grids = state[:, :, :self.grid_length].reshape(-1, 1, self.grid_side_size, self.grid_side_size)
        block_1 = state[:, :, self.grid_length:self.grid_length+self.block_length].reshape(-1, 1, self.block_side_size, self.block_side_size)
        block_2 = state[:, :, self.grid_length+self.block_length:self.grid_length+2*self.block_length].reshape(-1, 1, self.block_side_size, self.block_side_size)
        block_3 = state[:, :, -self.block_length:].reshape(-1, 1, self.block_side_size, self.block_side_size)

        # Encode grids and blocks using CNNs
        grid_enc = self.grid_encoder(grids)
        block_1_enc = self.block_encoder(block_1)
        block_2_enc = self.block_encoder(block_2)
        block_3_enc = self.block_encoder(block_3)

        # Bring encoded grids and blocks back into the same dimension
        grid_enc = grid_enc.reshape(batch_size, sequence_length, -1)
        block_1_enc = block_1_enc.reshape(batch_size, sequence_length, -1)
        block_2_enc = block_2_enc.reshape(batch_size, sequence_length, -1)
        block_3_enc = block_3_enc.reshape(batch_size, sequence_length, -1)

        # Concatenate encoded grids and blocks
        enc = torch.cat([grid_enc, block_1_enc, block_2_enc, block_3_enc], dim=-1)

        # Encode concatenated grids and blocks using MLP
        enc = self.res_mlp(enc)

        return enc



    







        

    