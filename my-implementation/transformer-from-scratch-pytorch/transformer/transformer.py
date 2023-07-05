"""Define transformer"""
import torch
import torch.nn as nn

from .encoders.encoder import TransformerEncoder
from .decoders.decoder import TransformerDecoder

class Transformer(nn.Module):
    """Transformer with self-attention mechanism"""

    def __init__(
            self,
            n_source_vocab: int,
            n_target_vocab: int,
            source_pad_idx: int,
            target_pad_idx: int,
            embed_dim: int,
            n_heads: int,
            ff_dim: int,
            n_layers: int = 6,
            drop_prob: float = 0.1,
            ):
        self.n_source_vocab = n_source_vocab
        self.n_target_vocab = n_target_vocab
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx 
        self.embed_dim = embed_dim

        self.encoder = TransformerEncoder(n_head = n_heads, 
                                          d_model = embed_dim, 
                                          n_layers = n_layers,
                                          )
        self.decoder = TransformerDecoder(n_head = n_heads, 
                                          d_model = embed_dim, 
                                          n_layers = n_layers,
                                          )


    def forward(self):
        pass