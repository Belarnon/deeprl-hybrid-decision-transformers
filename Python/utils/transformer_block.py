import numpy as np
import torch
from torch import nn

# code copied from exercise 7 of Deep Learning at ETH

class SelfAttention(nn.Module):
    """
    A SelfAttention model.
    
    Args:
        d: The embedding dimension.
        heads: The number of attention heads.
    """
    def __init__(self, d: int, n_head: int=8):
        super().__init__()
        self.h = n_head
        
        self.Wq = nn.Linear(d, d * n_head, bias=False)
        self.Wk = nn.Linear(d, d * n_head, bias=False)
        self.Wv = nn.Linear(d, d * n_head, bias=False)
        
        # This unifies the outputs of the different heads into 
        # a single k-dimensional vector.
        self.unifyheads = nn.Linear(n_head * d, d)
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            x: The input embedding of shape [b, l, d].
            attention_mask: The attention mask of shape [b, l, l].
            
        Returns:
            Self attention tensor of shape [b, l, d].
        """
        b, l, d = x.size()
        h = self.h
        
        # Transform the input embeddings x of shape [b, l, d] to queries, keys, values.
        # The output shape is [b, l, d*h] which we transform into [b, l, h, d]. Then,
        # we fold the heads into the batch dimenstion to arrive at [b*h, l, d]
        queries = self.Wq(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)
        keys = self.Wk(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)
        values = self.Wv(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)
        
        # Compute the product of queries and keys and scale with sqrt(d).
        # The tensor w' has shape (b*h, l, l) containing raw weights.
        #----------------
        w_prime = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d)
        #----------------

        # create zero-copy of w_prime to fill with attention mask
        # (b*h, l, l)
        w = torch.zeros_like(w_prime)

        # Apply the attention mask.
        if attention_mask is not None:
            
            # for each item in the batch, create the lower triangular matrix and pad with zeros
            for i, mask in enumerate(attention_mask):
                mask_len = int(mask.sum())
                pad_len = l - mask_len
                mask_matrix = mask.repeat(mask_len,1)[:, pad_len:]
                indices = torch.triu_indices(mask_len, mask_len, offset=1)
                mask_matrix[indices[0], indices[1]] = 0.0
                new_w = w_prime[i, pad_len:, pad_len:].clone()
                new_w[mask_matrix == 0] = float('-inf')
                sm_w = torch.nn.functional.softmax(new_w, dim=-1)
                w[i, pad_len:, pad_len:] = sm_w


            """# set every zero to -inf
            attention_matrix[attention_matrix == 0.0] = float('inf')
            # set upper offset triangular matrix to 0
            indices = torch.triu_indices(l,l, offset=1)
            attention_matrix[:, indices[0], indices[1]] = 0.0

            # copy again for each head
            attention_matrix = attention_mask.repeat_interleave(h, dim=0).float()
            # element-wise multiplication with weights
            w_prime = w_prime.mul(attention_matrix)"""

        else:
            # By default, only attend to the preceding elements.
            indices = torch.triu_indices(l, l, offset=1) # returns a tuple of two tensors specifying row and column indices
            # set to -inf to not break softmax!
            w_prime[:, indices[0], indices[1]] = float('-inf')
    
            # Compute w by normalizing w' over the last dimension.
            # Shape: [b*h, l, l]
            #----------------
            w = nn.functional.softmax(w_prime, dim=-1)
            #----------------
        
        
        # Apply the self attention to the values.
        # Shape: [b, h, l, d]
        #----------------
        out = torch.bmm(w, values).view(b, h, l, d)
        #----------------
        
        
        # Swap h, l back.
        # Shape: [b, l, h*d]
        out = out.transpose(1, 2).contiguous().view(b, l, h * d)
        
        # Unify heads to arrive at shape [b, l, d].
        return self.unifyheads(out)
    
class TransformerBlock(nn.Module):
    """
    A Transformer block consisting of self attention and ff-layer.
    
    Args:
        d (int): The embedding dimension.
        heads (int): The number of attention heads.
        n_mlp (int): The number of mlp 'blocks'.
    """
    def __init__(self, d: int, n_head: int=4, n_mlp: int=4):
        super().__init__()
        
        # The self attention layer.
        self.attention = SelfAttention(d, n_head=n_head)
        
        # The two layer norms.
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        
        # The feed-forward layer.
        self.ff = nn.Sequential(
            nn.Linear(d, n_mlp*d),
            nn.ReLU(),
            nn.Linear(n_mlp*d, d)
        )
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            x: The input embedding of shape [b, l, d].
            attention_mask: The attention mask of shape [b, l, l].
            
        Returns:
            Transformer output tensor of shape [b, l, d].
        """
        # Apply the self attention layer.
        out = self.attention(x, attention_mask) + x
        out = self.norm1(out)
        out = self.ff(out) + out
        out = self.norm2(out)

        return out
