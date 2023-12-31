import os
from typing import Iterable, List, Optional, Tuple

from faiss import Index
from langchain import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
import numpy as np

from .engine import Engine
from ..contracts import SearchResultEntity
from ..embeddings import Embedding

INDEX_FILE_NAME = 'index.faiss'
DATA_FILE_NAME = 'index.pkl'


class LangchainEmbedding(Embeddings):

    def __init__(self, embedding: Embedding):
        self.__embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return []

    def embed_query(self, text: str) -> List[float]:
        return self.__embedding.embed(text)


class LangChainEngine(Engine):

    def __init__(self, index: Index, embedding: Embedding):
        self.__index = index
        self.__embedding = embedding
        self.__init_langchain_faiss()

    def batch_insert_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None) -> None:
        self.__langchain_faiss.add_texts(texts, metadatas)

    def batch_insert_texts_with_embeddings(
            self,
            texts: Iterable[str],
            embeddings: Iterable[List[float]],
            metadatas: Optional[List[dict]] = None) -> None:
        if len(texts) != len(embeddings):
            raise ValueError('numbers of texts and embeddings do not match')
        count = len(embeddings)
        text_embeddings = [(texts[i], embeddings[i]) for i in range(count)]
        self.__langchain_faiss.add_embeddings(text_embeddings, metadatas)

    def search_by_text(self, query_text: str, top_k: int = 5) -> List[SearchResultEntity]:
        query_embedding = self.__embedding.embed(query_text)
        return self.search_by_embedding(query_embedding, top_k)

    def search_by_embedding(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResultEntity]:
        index_dimension = self.__langchain_faiss.index.d
        if len(query_embedding) != index_dimension:
            raise ValueError(f'query embedding dimension {len(query_embedding)}'
                             f' does not match index dimension {index_dimension}')
        docs = self.__langchain_faiss.similarity_search_with_score_by_vector(query_embedding, top_k)
        return self.__parse_docs(docs)

    def clear(self):
        self.__init_langchain_faiss()

    def remove(self, docstore_ids: Optional[List[str]]):
        """
        Function to remove documents from the vectorstore.
        
        Parameters
        ----------
        vectorstore : FAISS
            The vectorstore to remove documents from.
        docstore_ids : Optional[List[str]]
            The list of docstore ids to remove. If None, all documents are removed.
        
        Returns
        -------
        n_removed : int
            The number of documents removed.
        n_total : int
            The total number of documents in the vectorstore.
        
        Raises
        ------
        ValueError
            If there are duplicate ids in the list of ids to remove.
        """

        vectorstore = self.__langchain_faiss

        if docstore_ids is None:
            vectorstore.docstore = {}
            vectorstore.index_to_docstore_id = {}
            n_removed = vectorstore.index.ntotal
            n_total = vectorstore.index.ntotal
            vectorstore.index.reset()
            return n_removed, n_total
        
        set_ids = set(docstore_ids)
        if len(set_ids) != len(docstore_ids):
            raise ValueError("Duplicate ids in list of ids to remove.")
        index_ids = [
            i_id
            for i_id, d_id in vectorstore.index_to_docstore_id.items()
            if d_id in docstore_ids
        ]
        n_removed = len(index_ids)
        n_total = vectorstore.index.ntotal
        vectorstore.index.remove_ids(np.array(index_ids, dtype=np.int64))
        for i_id, d_id in zip(index_ids, docstore_ids):
            del vectorstore.docstore._dict[
                d_id
            ]  # remove the document from the docstore

            del vectorstore.index_to_docstore_id[
                i_id
            ]  # remove the index to docstore id mapping
        vectorstore.index_to_docstore_id = {
            i: d_id
            for i, d_id in enumerate(vectorstore.index_to_docstore_id.values())
        }
        return n_removed, n_total


    def merge_from(self, other_engine: 'LangChainEngine'):
        self.__langchain_faiss.merge_from(other_engine.__langchain_faiss)

    def load_data_index_from_disk(self, path: str):
        index_file = os.path.join(path, INDEX_FILE_NAME)
        data_file = os.path.join(path, DATA_FILE_NAME)

        if (not os.path.exists(index_file)) and (not os.path.exists(data_file)):
            self.__init_langchain_faiss()
        else:
            self.__langchain_faiss = FAISS.load_local(path, LangchainEmbedding(self.__embedding))

    def save_data_index_to_disk(self, path: str):
        self.__langchain_faiss.save_local(path)

    def get_store_files_size(self, path: str) -> int:
        index_file = os.path.join(path, INDEX_FILE_NAME)
        data_file = os.path.join(path, DATA_FILE_NAME)
        return os.path.getsize(index_file) + os.path.getsize(data_file)

    @staticmethod
    def get_index_file_relative_path():
        return INDEX_FILE_NAME

    @staticmethod
    def get_data_file_relative_path():
        return DATA_FILE_NAME

    def __init_langchain_faiss(self) -> FAISS:
        self.__index.reset()
        self.__langchain_faiss = FAISS(self.__embedding.embed, self.__index, InMemoryDocstore({}), {})

    @staticmethod
    def __parse_docs(docs: List[Tuple[Document, float]]) -> List[SearchResultEntity]:
        res = [SearchResultEntity(text=item[0].page_content,
                                  metadata=item[0].metadata, score=float(item[1])) for item in docs]
        return res

    def __len__(self):
        return self.__langchain_faiss.index.ntotal