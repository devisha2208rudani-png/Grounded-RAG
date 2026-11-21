from __future__ import annotations

"""Shared data models used across the RAG pipeline."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentSection:
    section_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceDocument:
    document_id: str
    title: str
    source_path: str
    sections: list[DocumentSection]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    token_count: int
    chunk_index: int
    citation_label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(**payload)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    dense_score: float
    lexical_score: float
    fused_score: float
    rerank_score: float = 0.0

    @property
    def final_score(self) -> float:
        return self.rerank_score or self.fused_score


@dataclass(slots=True)
class RetrievalResult:
    question: str
    results: list[RetrievedChunk]
    confidence: float
    low_confidence: bool


@dataclass(slots=True)
class Citation:
    citation_label: str
    title: str
    source_path: str
    snippet: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    fallback_used: bool
    retrieval: list[RetrievedChunk]


@dataclass(slots=True)
class EvaluationExample:
    query: str
    relevant_document_ids: list[str]
    expected_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationSummary:
    total_examples: int
    recall_at_k: float
    mrr: float
    citation_hit_rate: float
    term_coverage: float

