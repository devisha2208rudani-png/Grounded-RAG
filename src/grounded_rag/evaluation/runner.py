from __future__ import annotations

"""Evaluation runner for the RAG pipeline."""

import json
from pathlib import Path

from grounded_rag.evaluation.metrics import (
    citation_hit_rate,
    recall_at_k,
    reciprocal_rank,
    term_coverage,
)
from grounded_rag.models import EvaluationExample, EvaluationSummary
from grounded_rag.pipeline import RAGPipeline


def load_examples(dataset_path: Path) -> list[EvaluationExample]:
    return [
        EvaluationExample(**json.loads(line))
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_pipeline(pipeline: RAGPipeline, dataset_path: Path) -> EvaluationSummary:
    examples = load_examples(dataset_path)
    recall_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    citation_scores: list[float] = []
    term_scores: list[float] = []

    for example in examples:
        retrieval = pipeline.retrieve(example.query)
        predicted_document_ids = [result.chunk.document_id for result in retrieval.results]
        relevant_document_ids = set(example.relevant_document_ids)

        recall_scores.append(recall_at_k(relevant_document_ids, predicted_document_ids))
        reciprocal_ranks.append(reciprocal_rank(relevant_document_ids, predicted_document_ids))

        answer = pipeline.ask(example.query)
        cited_document_ids = [citation.metadata.get("document_id", "") for citation in answer.citations]
        citation_scores.append(citation_hit_rate(relevant_document_ids, cited_document_ids))
        term_scores.append(term_coverage(example.expected_terms, answer.answer))

    total = max(len(examples), 1)
    return EvaluationSummary(
        total_examples=len(examples),
        recall_at_k=sum(recall_scores) / total,
        mrr=sum(reciprocal_ranks) / total,
        citation_hit_rate=sum(citation_scores) / total,
        term_coverage=sum(term_scores) / total,
    )

