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


    def forward(self, encoder_input):
        """Encoder"""
        contextual_output = self.multi_head_attentions(encoder_input, encoder_input, encoder_input)
        x = self.norm1(encoder_input + contextual_output)
        feedforward = self.feedforward(x)
        encoder_output = self.norm2(feedforward + x)
        return encoder_output
    

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

    def forward(self, encoder_input):
        """transformer encoder block"""
        for layer in self.encoder_layers:
            # encoder's output will be the input for next encoder
            encoder_input = layer(encoder_input)
        return encoder_input