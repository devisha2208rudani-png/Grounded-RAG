from __future__ import annotations

"""Evaluation metrics for retrieval and grounded answer quality."""


def recall_at_k(relevant_document_ids: set[str], predicted_document_ids: list[str]) -> float:
    if not relevant_document_ids:
        return 0.0
    return 1.0 if any(document_id in relevant_document_ids for document_id in predicted_document_ids) else 0.0


def reciprocal_rank(relevant_document_ids: set[str], predicted_document_ids: list[str]) -> float:
    for rank, document_id in enumerate(predicted_document_ids, start=1):
        if document_id in relevant_document_ids:
            return 1.0 / rank
    return 0.0


def citation_hit_rate(relevant_document_ids: set[str], cited_document_ids: list[str]) -> float:
    if not relevant_document_ids:
        return 0.0
    return 1.0 if any(document_id in relevant_document_ids for document_id in cited_document_ids) else 0.0


def term_coverage(expected_terms: list[str], answer: str) -> float:
    if not expected_terms:
        return 1.0
    lowered_answer = answer.lower()
    hits = sum(1 for term in expected_terms if term.lower() in lowered_answer)
    return hits / len(expected_terms)

