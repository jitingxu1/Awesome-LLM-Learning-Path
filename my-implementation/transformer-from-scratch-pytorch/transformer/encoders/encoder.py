"""Define encoder"""
import torch
import torch.nn as nn

from ..attentions.multihead_attention import MultiHeadAttention

class Encoder(nn.Module):
    """Encoder with self-attention mechanism"""

    def __init__(
            self,
            n_head,
            d_model
            ):
         
        super().__init__()
        self.multi_head_attentions = MultiHeadAttention(n_head, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        # TODO: feedforward implementation
        self.feedforward = nn.Linear(d_model, d_model, bias=True) 
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, x):
        """Encoder"""
        attention = self.multi_head_attentions(x)
        x = self.norm1(x + attention)
        feedforward = self.fufeedforward(x)
        out = self.norm2(feedforward + x)
        return out
    

class TransformerEncoder(nn.Module):
    """Transformer encoder blocks"""

    def __init__(
            self,
            n_head,
            d_model,
            n_layers
            ):
        super().__init__()
        self.encoder_layers = nn.ModuleList([Encoder(n_head, d_model) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        """transformer encoder block"""
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        return x