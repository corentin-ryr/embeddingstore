import requests
from typing import List
from http import HTTPStatus

from ....core.contracts import SearchResultEntity
from ...contracts import StoreServiceConfig
from ...contracts.request_obj import CognitiveSearchRequestObj, CognitiveSearchVectorObj
from .agent import Agent

HEADER_API_KEY = 'api-key'
VALUE_FIELD_NAME = 'value'
SCORE_FIELD_NAME = '@search.score'


class CogSearchClient(Agent):

    def __init__(self, config: StoreServiceConfig):
        self.__config = config

    def load(self):
        return

    def search_by_embedding(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResultEntity]:

        url = self.__config.store_identifier

        headers = {}

        if self.__config.search_agent_api_key:
            headers[HEADER_API_KEY] = self.__config.search_agent_api_key.get_value()

        vector_obj = CognitiveSearchVectorObj(value=query_embedding, fields=self.__config.vector_field, k=top_k)
        vector_obj_list = []
        vector_obj_list.append(vector_obj.as_dict())
        request_obj = CognitiveSearchRequestObj(vectors=vector_obj_list)
        request_obj.update(self.__config.search_params)
        request_obj.update(self.__config.search_filters)

        response = requests.post(url=url, headers=headers, json=request_obj.as_dict(), verify=False)

        if response.status_code != HTTPStatus.OK:
            raise Exception(response.text)

        json_obj = response.json()
        target_list = json_obj[VALUE_FIELD_NAME]

        res = [SearchResultEntity(original_entity=item, score=item[SCORE_FIELD_NAME]) for item in target_list]

        if self.__config.vector_field is not None:
            for item in res:
                item.vector = item.original_entity[self.__config.vector_field]

        if self.__config.text_field is not None:
            for item in res:
                item.text = item.original_entity[self.__config.text_field]

        return res

    def clear(self):
        return
