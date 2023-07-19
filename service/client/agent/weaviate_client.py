from urllib import parse

import requests
from typing import List
from http import HTTPStatus
from ....core.contracts import SearchResultEntity
from ...contracts import StoreServiceConfig
from ...contracts.request_obj import WeaviateRequestObj
from .agent import Agent

ADDITIONAL_FIELD_NAME = "_additional"
SCORE_FIELD_NAME = "certainty"
VECTOR_FIELD_NAME = "vector"

RESPONSE_ERROR_FIELD = "errors"


class WeaviateClient(Agent):
    def __init__(self, config: StoreServiceConfig):
        self.__config = config
        if self.__config.store_identifier is None or self.__config.store_identifier == "":
            raise Exception("Empty store_identifier provided")
        elif self.__config.collection is None or self.__config.collection == "":
            raise Exception("Empty collection name provided")
        self.__graphql_url = parse.urljoin(self.__config.store_identifier, "v1/graphql")

    def load(self):
        return

    # TODO: support search_filters for Weaviate, transform to graphQL
    def search_by_embedding(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResultEntity]:

        class_name = self.__config.collection
        request_obj = WeaviateRequestObj(class_name=class_name, vector=query_embedding,
                                         text_field=self.__config.text_field, limit=top_k)
        headers = {"Content-type": "application/json"}
        if self.__config.search_agent_api_key.get_value() is not None:
            headers["Authorization"] = "Bearer " + self.__config.search_agent_api_key.get_value()
        response = requests.post(url=self.__graphql_url, headers=headers, json=request_obj.to_body())
        if response.status_code != HTTPStatus.OK:
            raise Exception(response.text)
        response_json = response.json()

        if RESPONSE_ERROR_FIELD in response_json and len(response_json[RESPONSE_ERROR_FIELD]) > 0:
            raise Exception(response_json[RESPONSE_ERROR_FIELD][0]["message"])

        target_list = response_json["data"]["Get"][class_name]
        res = [SearchResultEntity(original_entity=item, score=item[ADDITIONAL_FIELD_NAME][SCORE_FIELD_NAME],
                                  vector=item[ADDITIONAL_FIELD_NAME][VECTOR_FIELD_NAME]) for item in target_list]
        if self.__config.text_field is not None and self.__config.text_field != "":
            for item in res:
                item.text = item.original_entity[self.__config.text_field]
        return res

    def clear(self):
        return
