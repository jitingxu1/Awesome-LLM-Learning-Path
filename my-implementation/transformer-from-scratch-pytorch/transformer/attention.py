"""define the attention blocks"""
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int):
        super().__init__()

        self.n_head = n_head # number of attention heads
        self.d_model = d_model # dimensionality of the word embeddings and the hidden states in the model
        self.d_k = d_k # dimensionality of the keys
        self.d_v = d_v # dimensionality of the query

        # TODO: the relatinship between n_head and d_model

        # Initialize the learnable weights
        self.weight_query = nn.Linear(d_model, n_head * d_k, bias=False)
        self.weight_key = nn.Linear(d_model, n_head * d_k, bias=False)
        self.weight_value = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fullly_connected_layer = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self):
        """forward"""
        pass
