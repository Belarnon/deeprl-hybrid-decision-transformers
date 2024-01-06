import numpy as np
import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__(state_dim, act_dim, max_length)
    