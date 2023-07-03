"""modules"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """scaled dot product for similarity calculation"""

    def __init__(self):
        super().__init__()

    def forward(
            self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            mask: Optional[torch.Tensor] = None
            ):
        """implements the scaled dot product attention mechanism

        Parameters
        ----------
        query ``torch.Tensor``
            Represents the query tensor, 
            which has a shape of (batch_size, num_attention_heads, seq_length, hidden_size)
        key ``torch.Tensor``
            Represents the key tensor, 
            which has a shape of (batch_size, num_attention_heads, seq_length, hidden_size)
        value ``torch.Tensor``
            Represents the value tensor, 
            which has a shape of (batch_size, num_attention_heads, seq_length, d_v)
        mask ``Optional[torch.Tensor]``
            Represents the mask tensor, 
            used to mask certain elements in the attention calculation
            which has a shape of (batch_size, num_attention_heads, seq_length, d_v)

        Return
        -------
        values ``torch.Tensor``
            The contextualized representation obtained by multiplying the attention scores with the value tensor
            It has the same shape as the value tensor and represents the attended values for each position 
            in the input sequence. The values tensor has a shape of (batch_size, num_attention_heads, sequence_length, d_v).
        attentions ``torch.Tensor``
            Attention scores computed between the query and key tensors
             It has the same shape as the query and key tensors, 
             i.e., (batch_size, num_attention_heads, seq_length, seq_length). 
             The attention scores indicate the importance or weight assigned to 
             each value vector in the input sequence for each query vector.
        """


        # compute the attention score between query and all keys
        # The query and key tensor usually has a shape of 
        # (batch_size, num_attention_heads, seq_length, hidden_size).
        # attention result is (batch_size, num_heads, seq_length, seq_length)
        attention = torch.matmul(query, key.transpose(-2, -1))

        # It helps control the magnitude of the attention scores, 
        # preventing them from becoming too large.
        # Scaling by the square root of the hidden size 
        # (the last dimension of the query tensor) 
        # helps stabilize the attention distribution.
        attention = attention / math.sqrt(query.size()[-1])

        #The softmax operation ensures that the attention weights 
        # are non-negative and sum up to 1, 
        # indicating the relative importance or contribution 
        # of each key vector to the corresponding query vector.
        attention = F.softmax(attention, dim=-1)

        # mask certain elements in the attention tensor 
        # by replacing them with a specific value (-10000).
        if mask:
            attention = attention.masked_fill(mask == 0, -10000)

        # new contextualized representation
        # attention shape: (batch_size, num_attention_heads, sequence_length, sequence_length)
        # value shape: (batch_size, num_attention_heads, sequence_length, d_v)
        # values shape: (batch_size, num_attention_heads, sequence_length, d_v)
        values = torch.matmul(attention, value)

        return values, attention
