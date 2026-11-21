from __future__ import annotations

"""A lightweight BM25 retriever."""

import math
from collections import Counter

from grounded_rag.models import Chunk
from grounded_rag.utils.text import keyword_tokens


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.document_lengths: list[int] = []
        self.avg_doc_length = 0.0
        self._build()

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        query_terms = keyword_tokens(query)
        if not query_terms:
            return []

        scores: list[tuple[Chunk, float]] = []
        total_documents = len(self.chunks)
        for chunk, term_frequency, doc_length in zip(
            self.chunks,
            self.term_frequencies,
            self.document_lengths,
        ):
            score = 0.0
            for term in query_terms:
                tf = term_frequency.get(term, 0)
                if tf == 0:
                    continue
                df = self.document_frequencies.get(term, 0)
                idf = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1.0)
                )
                score += idf * numerator / denominator
            if score > 0:
                scores.append((chunk, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def _build(self) -> None:
        total_length = 0
        for chunk in self.chunks:
            corpus_text = " ".join(
                [
                    chunk.title,
                    str(chunk.metadata.get("section_title", "")),
                    chunk.text,
                ]
            )
            terms = keyword_tokens(corpus_text)
            term_frequency = Counter(terms)
            self.term_frequencies.append(term_frequency)
            self.document_lengths.append(len(terms))
            total_length += len(terms)
            for term in set(terms):
                self.document_frequencies[term] += 1
        self.avg_doc_length = total_length / max(len(self.chunks), 1)

