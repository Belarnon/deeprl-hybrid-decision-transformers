import torch

from modules.diag_gaussian_actor import SquashedNormal

class TenTenNLLActionLoss(torch.nn.Module):
    """
    Custom loss function for a 1010! action prediction task. This loss function
    is used for the online training of the actor network.
    """

    def __init__(self, block_selection_dim: int = 3, x_dim: int = 10):
        super().__init__()
        self.block_selection_dim = block_selection_dim
        self.x_dim = x_dim

    def partial_loss_fn(self, y_hat: SquashedNormal, y: torch.Tensor, entropy_regularization: float):
        """
        Args:
            y_hat (SquashedNormal): y_hat.sample() has shape (batch_size, seq_len, action_dim)
            y (torch.Tensor): (batch_size, seq_len, action_dim)

        Returns:
            loss (torch.Tensor): scalar
            log_likelihood (torch.Tensor): scalar
            entropy (torch.Tensor): scalar
        """
        
        log_likelihood = y_hat.log_likelihood(y).mean()
        entropy = y_hat.entropy().mean()
        loss = -log_likelihood - entropy_regularization * entropy

        return loss, -log_likelihood, entropy

    def forward(self, y_hat: SquashedNormal, y: torch.Tensor, entropy_regularization: float):
        """
        Use the log likelihood function to calculate the loss independently
        for each subsection of the action vector and add them together.
        The subsections are:

        1. Block selection
        2. X placement coordinate
        3. Y placement coordinate

        Formula: L = L_{choice} + L_{x} + L_{y}

        Args:
            y_hat (SquashedNormal): Predicted action distribution on shape (batch_size, seq_len, action_dim)
            y (torch.Tensor): Ground truth action, shape (batch_size, seq_len, action_dim)
            entropy_regularization (float): Entropy regularization coefficient (the current temperature)

        Returns:
            torch.Tensor: loss
        """

        selection_loss, selection_log_likelihood, selection_entropy = self._partial_log_likelihood(
            self._get_selection_vector(y_hat),
            self._get_selection_vector(y),
            entropy_regularization
        )

        x_loss, x_log_likelihood, x_entropy = self._partial_log_likelihood(
            self._get_x_vector(y_hat),
            self._get_x_vector(y),
            entropy_regularization
        )

        y_loss, y_log_likelihood, y_entropy = self._partial_log_likelihood(
            self._get_y_vector(y_hat),
            self._get_y_vector(y),
            entropy_regularization
        )

        return (selection_loss + x_loss + y_loss, 
                selection_log_likelihood + x_log_likelihood + y_log_likelihood,
                selection_entropy + x_entropy + y_entropy)

    
    def _get_selection_vector(self, y: torch.Tensor):
        """
        y: (batch_size, seq_len, action_dim)
        """
        return y[..., :self.block_selection_dim]
    
    def _get_x_vector(self, y: torch.Tensor):
        """
        y: (batch_size, seq_len, action_dim)
        """
        return y[..., self.block_selection_dim:self.block_selection_dim + self.x_dim]
    
    def _get_y_vector(self, y: torch.Tensor):
        """
        y: (batch_size, seq_len, action_dim)
        """
        return y[..., self.block_selection_dim + self.x_dim:]
