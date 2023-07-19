from dataclasses import dataclass

from promptflow.contracts.types import Secret
from promptflow.core.tools_manager import register_connections


@dataclass
class PineconeConnection:
    api_key: Secret
    api_base: str


register_connections(PineconeConnection)
