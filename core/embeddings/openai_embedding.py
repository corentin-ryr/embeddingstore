from typing import List

import openai

from .embedding import Embedding
from ..contracts import StoreCoreConfig, OpenAIApiType
from ..utils.retry_utils import retry_and_handle_exceptions


class OpenAIEmbedding(Embedding):

    @retry_and_handle_exceptions(exception_to_check=openai.error.RateLimitError, max_retries=5)
    def embed(self, text: str) -> List[float]:
        return openai.Embedding.create(
            input=text,
            model=self.__config.model_name)["data"][0]["embedding"]

    def __init__(self, config: StoreCoreConfig):
        self.__config = config

        openai.api_type = OpenAIApiType.OPENAI.value
        openai.api_base = config.model_api_base
        openai.api_version = config.model_api_version

        if config.model_api_key:
            openai.api_key = config.model_api_key.get_value()
