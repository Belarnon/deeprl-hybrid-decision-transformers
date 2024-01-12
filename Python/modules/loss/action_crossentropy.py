import torch

class TenTenActionLoss(torch.nn.Module):
    """
    Custom loss function for a 1010! action prediction task.
    """

    def __init__(self, block_selection_dim: int = 3, x_dim: int = 10):
        super().__init__()
        self.block_selection_dim = block_selection_dim
        self.x_dim = x_dim

        self.partial_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, y_hat, y):
        """
        y_hat: (batch_size, seq_len, action_dim)
        y: (batch_size, seq_len, action_dim)
        """

        # We want to calculate the loss independently for each subsection of the action
        # vector and add them together. The subsections are:
        # 1. Block selection
        # 2. X placement coordinate
        # 3. Y placement coordinate
        # Formula: L = L_{choice} + L_{x} + L_{y}

        selection_loss = self.partial_loss_fn(
            self._get_selection_vector(y_hat),
            self._get_selection_vector(y)
        )

        x_loss = self.partial_loss_fn(
            self._get_x_vector(y_hat),
            self._get_x_vector(y)
        )

        y_loss = self.partial_loss_fn(
            self._get_y_vector(y_hat),
            self._get_y_vector(y)
        )

        return selection_loss + x_loss + y_loss

    
    def _get_selection_vector(self, y):
        """
        y: (batch_size, seq_len, action_dim)
        """
        return y[:, :, :self.block_selection_dim]
    
    def _get_x_vector(self, y):
        """
        y: (batch_size, seq_len, action_dim)
        """
        return y[:, :, self.block_selection_dim:self.block_selection_dim + self.x_dim]
    
    def _get_y_vector(self, y):
        """
        y: (batch_size, seq_len, action_dim)
        """
        return y[:, :, self.block_selection_dim + self.x_dim:]
