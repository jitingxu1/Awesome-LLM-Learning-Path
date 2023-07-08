"""Define transformer"""
import torch
import torch.nn as nn

from .encoders.encoder import TransformerEncoder
from .decoders.decoder import TransformerDecoder
from .embeddings.transformer_embedding import TransformerEmbedding

class Transformer(nn.Module):
    """Transformer with self-attention mechanism"""

    def __init__(
            self,
            n_source_vocab: int,
            n_target_vocab: int,
            source_pad_idx: int,
            target_pad_idx: int,
            max_seq_len: int,
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

        self.embeddings = TransformerEmbedding(n_source_vocab, embed_dim, max_seq_len,padding_idx=source_pad_idx)

        self.encoder = TransformerEncoder(n_head = n_heads, 
                                          d_model = embed_dim, 
                                          n_layers = n_layers,
                                          )
        self.decoder = TransformerDecoder(n_head = n_heads, 
                                          d_model = embed_dim, 
                                          n_layers = n_layers,
                                          )
        self.linear = nn.Linear(embed_dim, ff_dim) # TODO: figure out the dimension
        self.softmax = nn.Softmax(self.n_target_vocab)

    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask


    def forward(self, encoder_input, decoder_input):
        """
        Args:
            encoder_input: input to encoder 
            decoder_input: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        embded_x = self.embeddings(encoder_input)
        encoder_output = self.encoder(embded_x)
        # TODO: mask function
        decoder_mask = self.make_trg_mask(decoder_input)
        decoder_output = self.decoder(encoder_output, decoder_input, decoder_mask)
        return self.softmax(self.linear(decoder_output))