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

        # logit transform to (0, 1)
        attention = F.softmax(attention, dim=-1)

        if mask:
            attention = attention.masked_fill(mask == 0, -10000)
        # Weighted sum of value vectors for each input token using attention scores -> 
        # new contextualized representation
        # (batch_size, num_heads, sequence_length, qkv_dim)
        values = torch.matmul(attention, v)

        return values, attention




