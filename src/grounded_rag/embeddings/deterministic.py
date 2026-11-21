from __future__ import annotations

"""Deterministic embedder used for tests and offline demos."""

import math

from grounded_rag.embeddings.base import EmbeddingModel
from grounded_rag.utils.text import tokenize


class DeterministicHashEmbedder(EmbeddingModel):
    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text, prefix="passage") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, prefix="query")

    def _embed(self, text: str, prefix: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(f"{prefix} {text}"):
            bucket = sum(ord(char) for char in token) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

