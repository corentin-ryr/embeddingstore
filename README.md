# Embedding store


*Embedding store* is a Python package by Microsoft to facilitate the synchronization of a [FAISS](https://github.com/facebookresearch/faiss) index on an Azure blob.

This fork of the Embedding Store Python package adds support for the FlatIP index that computes the Cosine Similarity instead of the L2 distance for the FlatL2 index already supported.
It also disables syncing every time an element is added to the index (for performance reasons).

## Installation

pip install git+https://github.com/corentin-ryr/embeddingstore
