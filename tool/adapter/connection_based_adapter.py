from typing import List
from .adapter import Adapter

from ...service.client.embeddingstore_client import EmbeddingStoreClient
from ..contracts import StoreType
from ..contracts.ui_config import VectorSearchToolUIConfig
from ..contracts.config import VectorSearchToolConfig
from promptflow.connections import CognitiveSearchConnection
from ...connections.pinecone import PineconeConnection
from ...connections.weaviate import WeaviateConnection
from ...connections.qdrant import QdrantConnection


class ConnectionBasedAdapter(Adapter):
    def __init__(self, ui_config: VectorSearchToolUIConfig):
        search_tool_config: VectorSearchToolConfig = None
        if isinstance(ui_config.connection, CognitiveSearchConnection):
            search_tool_config = self.__get_acs_config(ui_config)
        elif isinstance(ui_config.connection, PineconeConnection):
            search_tool_config = self.__get_pinecone_config(ui_config)
        elif isinstance(ui_config.connection, WeaviateConnection):
            search_tool_config = self.__get_weaviate_config(ui_config)
        elif isinstance(ui_config.connection, QdrantConnection):
            search_tool_config = self.__get_qdrant_config(ui_config)
        else:
            raise ValueError(f"Invalid connection type for vector db: {type(ui_config.connection)}")

        store_service_config = search_tool_config.generate_store_service_config()
        self.__store = EmbeddingStoreClient(store_service_config)

    def load(self):
        self.__store.load()

    def search(self, vector: List[float], top_k: int = 5) -> List[dict]:
        obj_list = self.__store.search_by_embedding(vector, top_k)
        return [obj.as_dict() for obj in obj_list]

    def __get_acs_config(self, ui_config: VectorSearchToolUIConfig):
        acs_connection: CognitiveSearchConnection = ui_config.connection
        return VectorSearchToolConfig(
            store_type=StoreType.COGNITIVESEARCH,
            url=(
                f"{acs_connection.api_base}/indexes/{ui_config.index_name}/"
                f"docs/search?api-version={acs_connection.api_version}"
            ),
            secret=acs_connection.api_key,
            vector_field=ui_config.vector_field,
            text_field=ui_config.text_field,
            search_params=ui_config.search_params,
            search_filters=ui_config.search_filters,
            logging_config=ui_config.logging_config
        )

    def __get_pinecone_config(self, ui_config: VectorSearchToolUIConfig):
        pinecone_connection: PineconeConnection = ui_config.connection
        return VectorSearchToolConfig(
            store_type=StoreType.PINECONE,
            url=pinecone_connection.api_base,
            collection=ui_config.namespace,
            secret=pinecone_connection.api_key,
            text_field=ui_config.text_field,
            search_filters=ui_config.search_filters,
            logging_config=ui_config.logging_config
        )

    def __get_weaviate_config(self, ui_config: VectorSearchToolUIConfig):
        weaviate_connection: WeaviateConnection = ui_config.connection
        return VectorSearchToolConfig(
            store_type=StoreType.WEAVIATE,
            url=weaviate_connection.api_base,
            collection=ui_config.class_name,
            secret=weaviate_connection.api_key,
            text_field=ui_config.text_field,
            logging_config=ui_config.logging_config
        )

    def __get_qdrant_config(self, ui_config: VectorSearchToolUIConfig):
        qdrant_connection: QdrantConnection = ui_config.connection
        return VectorSearchToolConfig(
            store_type=StoreType.QDRANT,
            url=qdrant_connection.api_base,
            collection=ui_config.collection_name,
            secret=qdrant_connection.api_key,
            text_field=ui_config.text_field,
            search_params=ui_config.search_params,
            search_filters=ui_config.search_filters,
            logging_config=ui_config.logging_config
        )
