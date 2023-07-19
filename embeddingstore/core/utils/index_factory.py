import faiss

from ..contracts import EmbeddingConfig
from ..contracts import IndexType


class IndexFactory:

    @staticmethod
    def get_index(config: EmbeddingConfig) -> faiss.Index:
        if config.index_type == IndexType.FLATL2:
            if config.dimension is None:
                return faiss.IndexFlatL2()
            return faiss.IndexFlatL2(config.dimension)
        if config.index_type == IndexType.FLATIP:
            if config.dimension is None:
                return faiss.IndexFlatIP()
            return faiss.IndexFlatIP(config.dimension)
        else:
            raise NotImplementedError("This has not been implemented yet.")
