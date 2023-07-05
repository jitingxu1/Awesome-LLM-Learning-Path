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
        # we have mask for the encoder output
        self.attention_with_mask = MultiHeadAttention(n_head, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        # self.transformer_encoder = TransformerEncoder(n_head, d_model, n_layers=1)
        self.attention = MultiHeadAttention(n_head, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self,
            encoder_output,
            decoder_input,
            mask):
        """decoder"""

        # TODO: need to handle the mask
        value = self.attention_with_mask(decoder_input, decoder_input, decoder_input, mask)
        value = self.norm1(value)
        out = self.attention(encoder_output, encoder_output, value)
        out = self.norm2(out+value) # skip connection
        out_linear = self.linear_layer(out)
        out = self.norm3(out+out_linear) #skip connection
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

    def forward(self, encoder_output, decoder_input, mask):
        """transformer decoder block"""
        for layer in self.decoder_layers:
            # each decoder's input
            # 1) decoder's output will be the input (q,k,v) for the first multihead attention
            # 2) encoder's output will be the query and key for the second multi-head attention
            # 3) output from first multihead-attention will be the value of second multihead         
            decoder_input = layer(encoder_output, decoder_input, mask)
        return decoder_input