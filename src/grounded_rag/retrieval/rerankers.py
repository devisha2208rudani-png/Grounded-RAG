from __future__ import annotations

"""Optional reranking strategies."""

from grounded_rag.models import RetrievedChunk
from grounded_rag.utils.text import keyword_tokens


class MetadataAwareReranker:
    def rerank(self, query: str, retrieved_chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ordered_query_terms = keyword_tokens(query)
        query_terms = set(ordered_query_terms)
        if not query_terms:
            return retrieved_chunks[:top_k]

        query_phrases = {
            " ".join(ordered_query_terms[index : index + window])
            for window in (2, 3)
            for index in range(0, max(len(ordered_query_terms) - window + 1, 0))
        }
        for item in retrieved_chunks:
            section_title = str(item.chunk.metadata.get("section_title", ""))
            chunk_terms = set(keyword_tokens(" ".join([item.chunk.text, item.chunk.title, section_title])))
            title_terms = set(keyword_tokens(item.chunk.title))
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
            section_overlap = len(query_terms & set(keyword_tokens(section_title))) / max(
                len(query_terms),
                1,
            )
            lowered_chunk_text = item.chunk.text.lower()
            phrase_boost = 1.0 if any(phrase in lowered_chunk_text for phrase in query_phrases) else 0.0
            item.rerank_score = (
                0.35 * item.fused_score
                + 0.40 * overlap
                + 0.10 * title_overlap
                + 0.05 * section_overlap
                + 0.10 * phrase_boost
            )
        return sorted(retrieved_chunks, key=lambda result: result.final_score, reverse=True)[:top_k]
