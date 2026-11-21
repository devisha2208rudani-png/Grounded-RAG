from __future__ import annotations

"""Prompt templates for grounded answer generation."""

from grounded_rag.models import RetrievedChunk


def build_context_block(retrieved_chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for item in retrieved_chunks:
        chunk = item.chunk
        page_number = chunk.metadata.get("page_number")
        section_title = chunk.metadata.get("section_title")
        location_bits = [bit for bit in [f"section={section_title}", f"page={page_number}"] if bit]
        location = ", ".join(location_bits) if location_bits else "section=body"
        blocks.append(
            "\n".join(
                [
                    f"[{chunk.citation_label}] title={chunk.title} | {location}",
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(blocks)


def build_messages(question: str, retrieved_chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    context = build_context_block(retrieved_chunks)
    system_prompt = (
        "You are a grounded assistant for document question answering.\n"
        "Use only the provided context.\n"
        "Cite supporting chunks inline using their citation labels, for example [architecture:2].\n"
        "If the evidence is incomplete or ambiguous, say so explicitly.\n"
        "Do not fabricate facts or sources."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Return a concise answer followed by a short 'Evidence' line listing the citations you used."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

