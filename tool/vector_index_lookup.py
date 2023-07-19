import uuid
from typing import List, Union

from promptflow import tool, ToolProvider
from promptflow.core.tools_manager import register_builtins

from .contracts.telemetry import StoreToolEventNames, StoreToolEventCustomDimensions
from .contracts.ui_config import StoreType, VectorSearchToolUIConfig
from .adapter import AdapterFactory
from .utils.logging import ToolLoggingUtils
from .utils.pf_runtime_utils import PromptflowRuntimeUtils
from ..core.contracts import StoreStage, StoreOperation
from ..core.logging.utils import LoggingUtils


class VectorIndexLookup(ToolProvider):

    def __init__(self, path: str):
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
                StoreToolEventCustomDimensions.STORE_TYPE: StoreType.MLINDEX
            }
        )

        super().__init__()

        ui_config = VectorSearchToolUIConfig(
            store_type=StoreType.MLINDEX,
            path=path,
            logging_config=logging_config
        )

        self.__adapter = AdapterFactory.get_adapter(ui_config)
        self.__adapter.load()

        self.__logger.telemetry_event_completed(
            event_name=StoreToolEventNames.INIT,
        )
        self.__logger.flush()

    @tool
    def search(
        self,
        query: Union[List[float], str],
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
            query: Union[List[float], str],
            top_k: int = 5
        ) -> List[dict]:
            return self.__adapter.search(query, top_k)

        return _do_search(query, top_k)


register_builtins(VectorIndexLookup)
