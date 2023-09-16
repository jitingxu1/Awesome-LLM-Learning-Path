import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """positional encodings"""

    def __init__(self, embed_dim: int, max_seq_len: int):
        super().__init__()
        self.embed_dim = embed_dim

        pos_embed = torch.zeros(max_seq_len, embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pos_embed[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pos_embed[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer('pos_embed', pos_embed)

    def forward(self, token_embeding: torch.Tensor):
        """add positional embedings to token embedings

        Parameters
        ----------
        token_embedings ``torch.Tensor``
            Token embedings, 
            which has a shape of (batch_size, seq_length, embeding_length)

        Return
        -------
        token_embeddings + positional embeddings ``torch.Tensor``
            Have the same shape to input 
        """
        # TODO: make embeding layer relative larger by multiplying the sqrt(self.embed_dim)

        token_embeding = token_embeding + self.pos_embed[:, : token_embeding.size(1)]
        return token_embeding
    