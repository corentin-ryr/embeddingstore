import logging
from embeddingstore.core.contracts import (
    StorageType,
    StoreCoreConfig,
    IndexType,
)
from embeddingstore.core.embeddingstore_core import EmbeddingStoreCore
import numpy as np
import time

DIMENSION = 1536

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    print("Creating FAISS store")

    # Configure an embedding store to store index file.
    blob_url = "https://devemsmiddlewarestorage.blob.core.windows.net/faiss-index-test"
    config = StoreCoreConfig.create_config(
        store_identifier=blob_url,
        storage_type=StorageType.BLOBSTORAGE,
        blob_conn_str="DefaultEndpointsProtocol=https;AccountName=devemsmiddlewarestorage;AccountKey=T6NN/OPURYXYoWeMFSTARb5e+yu8NXfHUJideMIBuzheU8h0ZwoBKc0F3cBsTIgQsRkqlHWs9OyAEfXR3OgdkA==;EndpointSuffix=core.windows.net",
        dimension=DIMENSION,
        create_if_not_exists=True,
        index_type=IndexType.FLATIP,
        local_cache_path="tmp"
    )
    storeContact = EmbeddingStoreCore(config)

    storeContact.clear()

    n = 1000
    #Generate 100 random texts
    print("Generating texts")
    texts = []
    for i in range(n):
        texts.append("text" + str(i))
    
    # Generate 100 random numpy arrays
    print("Generating embeddings")
    vectors = list(np.random.rand(n, DIMENSION))

    print("Inserting embeddings")
    startTime = time.time()
    storeContact.batch_insert_texts_with_embeddings(texts, vectors)
    print(f"Time taken to insert {n} embeddings: " + str(time.time() - startTime))

    time.sleep(5)

