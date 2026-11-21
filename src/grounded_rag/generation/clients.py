from __future__ import annotations

"""LLM clients and offline fallback generation."""

from typing import Protocol

from grounded_rag.exceptions import ConfigurationError, GenerationError
from grounded_rag.generation.prompts import build_messages
from grounded_rag.models import RetrievedChunk
from grounded_rag.utils.text import keyword_tokens, trim_snippet


class AnswerGenerator(Protocol):
    def generate(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        ...


class OpenAICompatibleGenerator:
    def __init__(
        self,
        model_name: str,
        api_key: str | None,
        base_url: str | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError("RAG_LLM_API_KEY is required to use the OpenAI-compatible generator.")
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GenerationError("The openai package is not installed.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = build_messages(question, retrieved_chunks)
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                messages=messages,
            )
        except Exception as exc:  # pragma: no cover - depends on external service
            raise GenerationError(f"LLM generation failed: {exc}") from exc
        return response.choices[0].message.content or ""


class RuleBasedGenerator:
    """Fallback generator that returns a grounded extractive summary with citations."""

    def generate(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "I do not have enough evidence in the indexed documents to answer that question."

        query_terms = set(keyword_tokens(question))
        evidence_lines: list[str] = []
        citations: list[str] = []

        for item in retrieved_chunks[:3]:
            chunk = item.chunk
            chunk_terms = set(keyword_tokens(chunk.text))
            overlap = query_terms & chunk_terms
            lead = trim_snippet(chunk.text, limit=180)
            if overlap:
                evidence_lines.append(f"- {lead} [{chunk.citation_label}]")
            else:
                evidence_lines.append(f"- Relevant context from {chunk.title}: {lead} [{chunk.citation_label}]")
            citations.append(f"[{chunk.citation_label}]")

        citations_text = ", ".join(citations)
        return (
            "I could not use a configured external LLM, so this answer is an extractive summary of the best-matching "
            f"chunks.\n\nKey evidence:\n{chr(10).join(evidence_lines)}\n\nEvidence: {citations_text}"
        )
