from __future__ import annotations

"""Dense vector retrieval index."""

import math

from grounded_rag.models import Chunk


class DenseVectorIndex:
    def __init__(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count must match.")
        self._items = list(zip(chunks, [self._normalize(vector) for vector in embeddings]))

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        normalized_query = self._normalize(query_vector)
        scored = [
            (chunk, self._dot_product(normalized_query, embedding))
            for chunk, embedding in self._items
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def serialize_embeddings(self) -> list[dict[str, object]]:
        return [
            {"chunk_id": chunk.chunk_id, "vector": embedding}
            for chunk, embedding in self._items
        ]

    @staticmethod
    def _dot_product(left: list[float], right: list[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

