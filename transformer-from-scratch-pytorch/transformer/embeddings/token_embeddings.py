from typing import Optional

from torch import nn

class TokenEmbedding(nn.Embedding):
    """
    Token emdeddings 
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            padding_idx: Optional[int]
            ):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param embed_dim: dimensions of model or the dimension of embeddings
        :param padding_idx
        """
        # several different ways to get embeddings for transfomrer
        # 1. initialize empty embeddings matrix and learn it during training
        # 2. user one or several pretrainined embeddings, 
        # we could refine or freeze it during training
        super(TokenEmbedding, self).__init__(vocab_size, embed_dim, padding_idx)
