from dataclasses import dataclass

from .types import StoreType
from ...core.contracts.config import LoggingConfig


@dataclass
class VectorSearchToolUIConfig:

    store_type: StoreType = None,

    path: str = None,  # for faiss and vector index

    connection: object = None,
    index_name: str = None,  # for cognitive search
    class_name: str = None,  # for weaviate search
    namespace: str = None,  # for pinecone search
    collection_name: str = None,  # for qdrant search
    text_field: str = None  # text field name in the response json from search engines
    vector_field: str = None  # vector field name in the response json from search engines
    search_params: dict = None  # additional params for making requests to search engines
    search_filters: dict = None  # additional filters for making requests to search engines

    logging_config: LoggingConfig = None
