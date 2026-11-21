from __future__ import annotations

"""Section-aware chunking for retrieval quality and source tracing."""

from dataclasses import dataclass

from grounded_rag.models import Chunk, SourceDocument
from grounded_rag.utils.text import sentence_split, tokenize, trim_snippet


@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


class SectionAwareChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk_document(self, document: SourceDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0

        for section in document.sections:
            section_title = section.metadata.get("section_title") or section.section_id
            section_chunks = self._chunk_text(section.text)
            for text in section_chunks:
                citation_label = f"{document.title}:{chunk_index}"
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}-chunk-{chunk_index}",
                        document_id=document.document_id,
                        title=document.title,
                        text=text,
                        token_count=len(tokenize(text)),
                        chunk_index=chunk_index,
                        citation_label=citation_label,
                        metadata={
                            "source_path": document.source_path,
                            "section_id": section.section_id,
                            "section_title": section_title,
                            **section.metadata,
                        },
                    )
                )
                chunk_index += 1
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        sentences = sentence_split(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = tokenize(sentence)
            if len(sentence_tokens) > self.config.chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences).strip())
                    current_sentences = []
                    current_tokens = 0
                chunks.extend(self._split_long_sentence(sentence_tokens))
                continue

            exceeds_limit = current_sentences and (
                current_tokens + len(sentence_tokens) > self.config.chunk_size
            )
            if exceeds_limit:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = self._tail_overlap(current_sentences)
                current_tokens = sum(len(tokenize(item)) for item in current_sentences)

            current_sentences.append(sentence)
            current_tokens += len(sentence_tokens)

        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())

        return [trim_snippet(chunk, limit=5000) if len(chunk) > 5000 else chunk for chunk in chunks]

    def _tail_overlap(self, sentences: list[str]) -> list[str]:
        if self.config.chunk_overlap <= 0:
            return []
        overlap: list[str] = []
        running_tokens = 0
        for sentence in reversed(sentences):
            sentence_tokens = len(tokenize(sentence))
            if overlap and running_tokens + sentence_tokens > self.config.chunk_overlap:
                break
            overlap.insert(0, sentence)
            running_tokens += sentence_tokens
        return overlap

    def _split_long_sentence(self, tokens: list[str]) -> list[str]:
        chunks: list[str] = []
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.config.chunk_size]
            if not window:
                continue
            chunks.append(" ".join(window))
            if start + self.config.chunk_size >= len(tokens):
                break
        return chunks

