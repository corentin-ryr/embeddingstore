import uuid
from typing import List, Optional, Union

from promptflow import tool, ToolProvider
from promptflow.core.tools_manager import register_builtins
from promptflow.connections import CognitiveSearchConnection

from .contracts.types import StoreType
from .contracts.telemetry import StoreToolEventNames, StoreToolEventCustomDimensions
from .contracts.ui_config import VectorSearchToolUIConfig
from ..core.utils.retry_utils import retry_and_handle_exceptions
from .adapter import AdapterFactory
from .utils.logging import ToolLoggingUtils
from .utils.pf_runtime_utils import PromptflowRuntimeUtils
from ..service.contracts.errors import EmbeddingSearchRetryableError
from ..core.contracts import StoreStage, StoreOperation
from ..core.logging.utils import LoggingUtils
from ..connections.pinecone import PineconeConnection
from ..connections.weaviate import WeaviateConnection
from ..connections.qdrant import QdrantConnection


class VectorDBLookup(ToolProvider):

    def __init__(
        self,
        connection: Union[CognitiveSearchConnection, PineconeConnection, WeaviateConnection, QdrantConnection],
        index_name: Optional[str] = None,
        class_name: Optional[str] = None,
        namespace: Optional[str] = None,
        collection_name: Optional[str] = None,
        text_field: Optional[str] = None,
        vector_field: Optional[str] = None,
        search_params: Optional[dict] = None,
        search_filters: Optional[dict] = None
    ):
        logging_config = ToolLoggingUtils.generate_config(
            tool_name=self.__class__.__name__
        )
        self.__logger = LoggingUtils.sdk_logger(__package__, logging_config)
        self.__logger.update_telemetry_context(
            {
                StoreToolEventCustomDimensions.TOOL_INSTANCE_ID: str(uuid.uuid4())
            }
        )

        self.__logger.telemetry_event_started(
            event_name=StoreToolEventNames.INIT,
            store_stage=StoreStage.INITIALIZATION,
            custom_dimensions={
                StoreToolEventCustomDimensions.STORE_TYPE: StoreType.DBSERVICE
            }
        )

        super().__init__()

        ui_config = VectorSearchToolUIConfig(
            store_type=StoreType.DBSERVICE,
            connection=connection,
            index_name=index_name,
            class_name=class_name,
            namespace=namespace,
            collection_name=collection_name,
            text_field=text_field,
            vector_field=vector_field,
            search_filters=search_filters,
            search_params=search_params,
            logging_config=logging_config
        )

        self.__adapter = AdapterFactory.get_adapter(ui_config)
        self.__adapter.load()

        self.__logger.telemetry_event_completed(
            event_name=StoreToolEventNames.INIT
        )
        self.__logger.flush()

    @tool
    @retry_and_handle_exceptions(EmbeddingSearchRetryableError)
    def search(
        self,
        vector: List[float],
        top_k: int = 5
    ) -> List[dict]:

        pf_context = PromptflowRuntimeUtils.get_pf_context_info_for_telemetry()

        @LoggingUtils.log_event(
            package_name=__package__,
            event_name=StoreToolEventNames.SEARCH,
            scope_context=pf_context,
            store_stage=StoreStage.SEARVING,
            store_operation=StoreOperation.SEARCH,
            logger=self.__logger,
            flush=True
        )
        def _do_search(
            vector: List[float],
            top_k: int = 5
        ) -> List[dict]:
            return self.__adapter.search(vector, top_k)

        return _do_search(vector, top_k)


register_builtins(VectorDBLookup)
