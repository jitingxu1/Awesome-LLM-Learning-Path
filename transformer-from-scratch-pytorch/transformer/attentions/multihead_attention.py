"""define the attention blocks"""
import torch
import torch.nn as nn

from .scaled_dot_product import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""

    def __init__(self, n_head: int, d_model: int):
        super().__init__()

        self.n_head = n_head # number of attention heads
        self.d_model = d_model # dimensionality of the word embeddings and the hidden states in the model

        assert self.n_model % self.n_head == 0, "Make sure evenly splitting"

        # Initialize the learnable weights
        self.weight_query = nn.Linear(d_model, d_model, bias=False) 
        self.weight_key = nn.Linear(d_model, d_model, bias=False)
        self.weight_value = nn.Linear(d_model, d_model, bias=False)
        self.scaled_dot_product_attention = ScaledDotProductAttention()
        self.fullly_connected_layer = nn.Linear(d_model, d_model, bias=False)
      

    def forward(self, q, k, v, mask=None):
        """forward"""
        
        # Convert input into queries, keys, and values
        # by using trainable layers
        q = self.weight_query(q)
        k = self.weight_key(k)
        v = self.weight_value(v)

        # Split weights into n_heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # Calculate the scaled dot product
        values, _ = self.scaled_dot_product_attention(q, k, v, mask)

        # Concat 
        out = self.concat(values)

        # Fully connected layer
        out = self.fullly_connected_layer(out)

        return out
    
    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor
    
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor