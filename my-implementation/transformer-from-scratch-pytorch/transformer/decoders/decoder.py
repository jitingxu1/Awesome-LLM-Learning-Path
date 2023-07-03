"""Define decoder"""
import torch
import torch.nn as nn

from ..attentions.multihead_attention import MultiHeadAttention
from ..encoders.encoder import TransformerEncoder

class Decoder(nn.Module):
    """Decoder with self-attention mechanism"""

    def __init__(
            self,
            n_head,
            d_model
            ):
        super().__init__()
        self.attention_with_mask = MultiHeadAttention(n_head, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(n_head, d_model, n_layers=1)

    def forward(self,
            encoder_output,
            decoder_input,
            mask):
        """decoder"""
        value = self.attention_with_mask(decoder_input, decoder_input, decoder_input, mask)
        value = self.norm(value)
        out = self.transformer_encoder(encoder_output, encoder_output, value)
        return out

class TransformerDecoder(nn.Module):
    """Transformer decoder block"""

    def __init__(
            self,
            n_head,
            d_model,
            n_layers
            ):
        super().__init__()

        self.decode_layers = nn.ModuleList(
            [
                Decoder(n_head, d_model) for _ in range(n_layers)
            ]
        )  

    def forward(self, x, mask):
        """transformer decoder block"""
        for layer in self.decoder_layers:
            x = layer(x, mask)
        return x  