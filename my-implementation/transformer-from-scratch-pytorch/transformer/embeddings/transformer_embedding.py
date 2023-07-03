from typing import Optional

from torch import nn

from .token_embeddings import TokenEmbedding
from .positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """
    Transformer emdeddings: 
    """

    def __init__(
            self,
            n_vocab: int,
            embed_dim: int, # d_model
            max_seq_len: int,
            padding_idx: Optional[int],
            drop_prob: Optional[float] = 0.1
            ):
        self.token_embeddings = TokenEmbedding(n_vocab, embed_dim, padding_idx)
        self.positional_embeddings = PositionalEncoding(embed_dim, max_seq_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """transformer embedding"""
        x = self.token_embeddings(x)
        x = self.positional_embeddings(x) # added the token_embeddings inside
        x = self.drop_out(x)
        return x

