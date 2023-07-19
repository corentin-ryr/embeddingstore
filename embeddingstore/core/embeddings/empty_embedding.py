from typing import List

from .embedding import Embedding


class EmptyEmbedding(Embedding):

    def embed(self, text: str) -> List[float]:
        raise ValueError('built-in embedding model or customized embedding function need to be specified')

    def __init__(self):
        return
