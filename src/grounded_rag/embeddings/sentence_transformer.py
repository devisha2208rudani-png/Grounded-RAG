from __future__ import annotations

"""SentenceTransformer-backed embedding generation with caching."""

import logging

from grounded_rag.cache import SQLiteCache
from grounded_rag.embeddings.base import EmbeddingModel
from grounded_rag.utils.text import stable_hash

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(EmbeddingModel):
    def __init__(self, model_name: str, cache: SQLiteCache | None = None) -> None:
        self.model_name = model_name
        self.cache = cache
        self._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._prefix_for_documents(text) for text in texts]
        return self._embed_batch(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([self._prefix_for_query(text)])[0]

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install project dependencies to enable dense embeddings."
                ) from exc
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(texts):
            cache_key = stable_hash(f"{self.model_name}:{text}")
            if self.cache:
                cached = self.cache.get("embeddings", cache_key)
                if cached is not None:
                    vectors[index] = cached
                    continue
            missing_indices.append(index)
            missing_texts.append(text)

        if missing_texts:
            model = self._load_model()
            generated_vectors = model.encode(
                missing_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for index, vector in zip(missing_indices, generated_vectors):
                dense_vector = [float(value) for value in vector]
                vectors[index] = dense_vector
                if self.cache:
                    cache_key = stable_hash(f"{self.model_name}:{texts[index]}")
                    self.cache.set("embeddings", cache_key, dense_vector)

        return [vector for vector in vectors if vector is not None]

    def _prefix_for_documents(self, text: str) -> str:
        return f"passage: {text}"

    def _prefix_for_query(self, text: str) -> str:
        return f"query: {text}"

