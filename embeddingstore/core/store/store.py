from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from ..contracts import SearchResultEntity


class Store(ABC):

    @abstractmethod
    def batch_insert_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None) -> None:
        pass

    @abstractmethod
    def batch_insert_texts_with_embeddings(
            self,
            texts: Iterable[str],
            embeddings: Iterable[List[float]],
            metadatas: Optional[List[dict]] = None) -> None:
        pass

    @abstractmethod
    def search_by_text(self, query_text: str, top_k: int = 5) -> List[SearchResultEntity]:
        pass

    @abstractmethod
    def search_by_embedding(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResultEntity]:
        pass

    @abstractmethod
    def merge_from(self, other_store: 'Store'):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def remove(self, docstore_ids: Optional[List[str]]):
        pass

    @abstractmethod
    def save(self):
        pass
