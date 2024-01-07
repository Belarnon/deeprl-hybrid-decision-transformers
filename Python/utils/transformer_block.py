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
    def __init__(self, d: int, heads: int=8):
        super().__init__()
        self.h = heads
        
        self.Wq = nn.Linear(d, d * heads, bias=False)
        self.Wk = nn.Linear(d, d * heads, bias=False)
        self.Wv = nn.Linear(d, d * heads, bias=False)
        
        # This unifies the outputs of the different heads into 
        # a single k-dimensional vector.
        self.unifyheads = nn.Linear(heads * d, d)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input embedding of shape [b, l, d].
            
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
    def __init__(self, d: int, heads: int=8, n_mlp: int=4):
        super().__init__()
        
        # The self attention layer.
        self.attention = SelfAttention(d, heads=heads)
        
        # The two layer norms.
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        
        # The feed-forward layer.
        self.ff = nn.Sequential(
            nn.Linear(d, n_mlp*d),
            nn.ReLU(),
            nn.Linear(n_mlp*d, d)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input embedding of shape [b, l, d].
            
        Returns:
            Transformer output tensor of shape [b, l, d].
        """
        # Implement the forward pass as shown in the figure above.
        #----------------
        out = self.attention(x) + x
        out = self.norm1(out)
        out = self.ff(out) + out
        out = self.norm2(out)
        #----------------
        return out
