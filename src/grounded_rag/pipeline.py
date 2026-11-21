from __future__ import annotations

"""Main orchestration pipeline for indexing and question answering."""

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from grounded_rag.cache import SQLiteCache
from grounded_rag.config import Settings
from grounded_rag.embeddings.base import EmbeddingModel
from grounded_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from grounded_rag.generation.clients import (
    AnswerGenerator,
    OpenAICompatibleGenerator,
    RuleBasedGenerator,
)
from grounded_rag.index.dense import DenseVectorIndex
from grounded_rag.index.repository import IndexRepository
from grounded_rag.ingestion.chunker import ChunkingConfig, SectionAwareChunker
from grounded_rag.ingestion.loaders import DocumentLoader
from grounded_rag.ingestion.pipeline import IngestionPipeline
from grounded_rag.logging_utils import configure_logging
from grounded_rag.models import AnswerResult, Chunk, Citation, RetrievalResult, RetrievedChunk
from grounded_rag.retrieval.hybrid import HybridRetriever
from grounded_rag.retrieval.lexical import BM25Retriever
from grounded_rag.retrieval.rerankers import MetadataAwareReranker
from grounded_rag.utils.text import clean_text, stable_hash, trim_snippet

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingModel | None = None,
        answer_generator: AnswerGenerator | None = None,
        cache: SQLiteCache | None = None,
        repository: IndexRepository | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        configure_logging(self.settings.log_level)

        self.cache = cache or SQLiteCache(self.settings.cache_path)
        self.repository = repository or IndexRepository(self.settings.artifact_dir)
        self.embedder = embedder or self._build_embedder()
        self.answer_generator = answer_generator or self._build_generator()
        self.ingestion_pipeline = IngestionPipeline(
            loader=DocumentLoader(),
            chunker=SectionAwareChunker(
                ChunkingConfig(
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
            ),
        )
        self.reranker = MetadataAwareReranker() if self.settings.enable_rerank else None
        self._dense_index: DenseVectorIndex | None = None
        self._lexical_retriever: BM25Retriever | None = None
        self._manifest: dict[str, Any] = {}

    def build_index(self, source_dir: Path | None = None) -> dict[str, Any]:
        source_dir = source_dir or self.settings.source_dir
        documents, chunks = self.ingestion_pipeline.ingest_directory(source_dir)
        embeddings = self.embedder.embed_documents([self._embedding_text(chunk) for chunk in chunks])

        self._dense_index = DenseVectorIndex(chunks, embeddings)
        self._lexical_retriever = BM25Retriever(chunks)
        self._manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(source_dir),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "embedding_model": getattr(self.embedder, "model_name", type(self.embedder).__name__),
            "settings": {
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "top_k": self.settings.top_k,
                "candidate_pool_size": self.settings.candidate_pool_size,
                "hybrid_alpha": self.settings.hybrid_alpha,
                "rerank_enabled": self.settings.enable_rerank,
            },
        }
        self.repository.save(chunks=chunks, embeddings=embeddings, metadata=self._manifest)
        logger.info("Persisted index with %s chunks to %s", len(chunks), self.settings.artifact_dir)
        return self._manifest

    def load_index(self) -> dict[str, Any]:
        persisted = self.repository.load()
        self._dense_index = DenseVectorIndex(persisted.chunks, persisted.embeddings)
        self._lexical_retriever = BM25Retriever(persisted.chunks)
        self._manifest = persisted.metadata
        logger.info("Loaded persisted index with %s chunks", len(persisted.chunks))
        return self._manifest

    def ensure_index(self) -> dict[str, Any]:
        if self._dense_index and self._lexical_retriever:
            return self._manifest
        try:
            return self.load_index()
        except Exception:
            logger.info("No usable index found on disk. Building a fresh index.")
            return self.build_index()

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        normalized_question = clean_text(question)
        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        self.ensure_index()
        if not self._dense_index or not self._lexical_retriever:
            raise RuntimeError("Retrieval components were not initialized.")

        query_vector = self.embedder.embed_query(normalized_question)
        retriever = HybridRetriever(
            dense_index=self._dense_index,
            lexical_retriever=self._lexical_retriever,
            hybrid_alpha=self.settings.hybrid_alpha,
            confidence_threshold=self.settings.confidence_threshold,
        )
        retrieval = retriever.search(
            question=normalized_question,
            query_vector=query_vector,
            top_k=top_k or self.settings.top_k,
            candidate_pool_size=self.settings.candidate_pool_size,
        )
        if self.reranker and retrieval.results:
            retrieval.results = self.reranker.rerank(
                normalized_question,
                retrieval.results,
                top_k or self.settings.top_k,
            )
        return retrieval

    def ask(self, question: str, top_k: int | None = None) -> AnswerResult:
        normalized_question = clean_text(question)
        self.ensure_index()
        cache_key = stable_hash(
            f"{normalized_question}:{top_k or self.settings.top_k}:{self._manifest.get('created_at', 'na')}"
        )
        cached = self.cache.get("responses", cache_key)
        if cached is not None:
            return self._answer_from_cache(cached)

        retrieval = self.retrieve(normalized_question, top_k=top_k)
        citations = self._build_citations(retrieval.results)

        if retrieval.low_confidence:
            answer_text = (
                "I do not have enough grounded evidence in the indexed documents to answer confidently. "
                "Try narrowing the question or ingesting more relevant material."
            )
            fallback_used = True
        else:
            answer_text = self.answer_generator.generate(normalized_question, retrieval.results)
            fallback_used = False

        answer = AnswerResult(
            question=normalized_question,
            answer=answer_text,
            citations=citations,
            confidence=retrieval.confidence,
            fallback_used=fallback_used,
            retrieval=retrieval.results,
        )
        self.cache.set("responses", cache_key, self._answer_to_cache(answer))
        return answer

    def _build_embedder(self) -> EmbeddingModel:
        return SentenceTransformerEmbedder(
            model_name=self.settings.embedding_model,
            cache=self.cache,
        )

    def _embedding_text(self, chunk: Chunk) -> str:
        section_title = str(chunk.metadata.get("section_title", ""))
        return " ".join(part for part in [chunk.title, section_title, chunk.text] if part)

    def _build_generator(self) -> AnswerGenerator:
        if self.settings.llm_api_key:
            return OpenAICompatibleGenerator(
                model_name=self.settings.llm_model,
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
            )
        return RuleBasedGenerator()

    def _build_citations(self, retrieved_chunks: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for item in retrieved_chunks:
            chunk = item.chunk
            citations.append(
                Citation(
                    citation_label=chunk.citation_label,
                    title=chunk.title,
                    source_path=str(chunk.metadata.get("source_path", "")),
                    snippet=trim_snippet(chunk.text),
                    score=item.final_score,
                    metadata={
                        "document_id": chunk.document_id,
                        "section_title": chunk.metadata.get("section_title"),
                        "page_number": chunk.metadata.get("page_number"),
                    },
                )
            )
        return citations

    def _answer_to_cache(self, answer: AnswerResult) -> dict[str, Any]:
        return {
            "question": answer.question,
            "answer": answer.answer,
            "citations": [asdict(citation) for citation in answer.citations],
            "confidence": answer.confidence,
            "fallback_used": answer.fallback_used,
            "retrieval": [
                {
                    "chunk": item.chunk.to_dict(),
                    "dense_score": item.dense_score,
                    "lexical_score": item.lexical_score,
                    "fused_score": item.fused_score,
                    "rerank_score": item.rerank_score,
                }
                for item in answer.retrieval
            ],
        }

    def _answer_from_cache(self, payload: dict[str, Any]) -> AnswerResult:
        return AnswerResult(
            question=payload["question"],
            answer=payload["answer"],
            citations=[Citation(**citation) for citation in payload["citations"]],
            confidence=float(payload["confidence"]),
            fallback_used=bool(payload["fallback_used"]),
            retrieval=[
                RetrievedChunk(
                    chunk=Chunk.from_dict(item["chunk"]),
                    dense_score=float(item["dense_score"]),
                    lexical_score=float(item["lexical_score"]),
                    fused_score=float(item["fused_score"]),
                    rerank_score=float(item["rerank_score"]),
                )
                for item in payload["retrieval"]
            ],
        )

    def close(self) -> None:
        self.cache.close()
