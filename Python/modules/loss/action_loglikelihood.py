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

    def forward(self, y_hat: SquashedNormal, y: torch.Tensor, attention_mask: torch.Tensor, entropy_regularization: float):
        """
        Use the log likelihood function to calculate the loss *for all actions together*,
        in contrast to the CrossEntropyLoss for the offline training, where we calculate
        the partial loss for each action separately and then sum them up. According to Kevin,
        we can do this here xD 

        Args:
            y_hat (SquashedNormal): Predicted action distribution on shape (batch_size, seq_len, action_dim)
            y (torch.Tensor): Ground truth action, shape (batch_size, seq_len, action_dim)
            attention_mask (torch.Tensor): Attention mask for the transformer (attended states are 0, padded states are 1), shape (batch_size, seq_len)
            entropy_regularization (float): Entropy regularization coefficient (the current temperature)

        Returns:
            loss (torch.Tensor): scalar
            log_likelihood (torch.Tensor): scalar
            entropy (torch.Tensor): scalar
        """

        a = y_hat.log_likelihood(y)
        b = a[attention_mask < 1]
        c = b.mean()

        log_likelihood = y_hat.log_likelihood(y)[attention_mask < 1].mean()
        entropy = y_hat.entropy().mean()
        loss = -log_likelihood - entropy_regularization * entropy

        return loss, -log_likelihood, entropy
