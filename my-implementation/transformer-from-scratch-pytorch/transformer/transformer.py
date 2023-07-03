"""Define transformer"""
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):

    def __init__(
            self,
            n_heads: int,
            embed_dim: int,
            n_layer: int,
            ff_dim: int,
            drop_prob: float
            ):
        super().__init__()
        

    def forward(self):
        pass

class TransformerDecoder(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass

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

        self.encoder = TransformerEncoder(n_heads = n_heads, 
                                          embed_dim = embed_dim, 
                                          n_layers = n_layers,
                                          ff_dim = ff_dim,
                                          drop_prob = drop_prob
                                          )
        self.decoder = TransformerDecoder(n_heads = n_heads, 
                                          embed_dim = embed_dim, 
                                          n_layers = n_layers,
                                          ff_dim = ff_dim,
                                          drop_prob = drop_prob
                                          )


    def forward(self):
        pass