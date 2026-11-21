from __future__ import annotations

"""Hybrid retrieval combining dense and lexical relevance signals."""

from grounded_rag.index.dense import DenseVectorIndex
from grounded_rag.models import Chunk, RetrievalResult, RetrievedChunk
from grounded_rag.retrieval.lexical import BM25Retriever


class HybridRetriever:
    def __init__(
        self,
        dense_index: DenseVectorIndex,
        lexical_retriever: BM25Retriever,
        hybrid_alpha: float,
        confidence_threshold: float,
    ) -> None:
        self.dense_index = dense_index
        self.lexical_retriever = lexical_retriever
        self.hybrid_alpha = hybrid_alpha
        self.confidence_threshold = confidence_threshold

    def search(
        self,
        question: str,
        query_vector: list[float],
        top_k: int,
        candidate_pool_size: int,
    ) -> RetrievalResult:
        dense_results = self.dense_index.search(query_vector, candidate_pool_size)
        lexical_results = self.lexical_retriever.search(question, candidate_pool_size)

        dense_scores = {chunk.chunk_id: score for chunk, score in dense_results}
        lexical_scores = {chunk.chunk_id: score for chunk, score in lexical_results}
        dense_ranks = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(dense_results, start=1)}
        lexical_ranks = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(lexical_results, start=1)}
        chunks_by_id: dict[str, Chunk] = {
            chunk.chunk_id: chunk for chunk, _ in dense_results + lexical_results
        }

        dense_max = max(dense_scores.values(), default=1.0)
        lexical_max = max(lexical_scores.values(), default=1.0)
        combined_results: list[RetrievedChunk] = []

        for chunk_id in chunks_by_id:
            dense_score = dense_scores.get(chunk_id, 0.0)
            lexical_score = lexical_scores.get(chunk_id, 0.0)
            dense_norm = dense_score / dense_max if dense_max else 0.0
            lexical_norm = lexical_score / lexical_max if lexical_max else 0.0
            reciprocal_rank_boost = 0.0
            if chunk_id in dense_ranks:
                reciprocal_rank_boost += 1.0 / (60 + dense_ranks[chunk_id])
            if chunk_id in lexical_ranks:
                reciprocal_rank_boost += 1.0 / (60 + lexical_ranks[chunk_id])
            fused_score = (
                self.hybrid_alpha * dense_norm
                + (1 - self.hybrid_alpha) * lexical_norm
                + reciprocal_rank_boost
            )
            combined_results.append(
                RetrievedChunk(
                    chunk=chunks_by_id[chunk_id],
                    dense_score=dense_score,
                    lexical_score=lexical_score,
                    fused_score=fused_score,
                )
            )

        combined_results.sort(key=lambda item: item.fused_score, reverse=True)
        top_results = combined_results[:top_k]
        confidence = self._estimate_confidence(top_results)
        return RetrievalResult(
            question=question,
            results=top_results,
            confidence=confidence,
            low_confidence=confidence < self.confidence_threshold,
        )

    def _estimate_confidence(self, results: list[RetrievedChunk]) -> float:
        if not results:
            return 0.0
        top = results[0]
        second = results[1] if len(results) > 1 else None
        dense_component = max(top.dense_score, 0.0)
        lexical_component = top.lexical_score / (1.0 + top.lexical_score)
        score_gap = max(top.fused_score - (second.fused_score if second else 0.0), 0.0)
        confidence = 0.55 * dense_component + 0.35 * lexical_component + 0.10 * score_gap
        return min(max(confidence, 0.0), 1.0)
