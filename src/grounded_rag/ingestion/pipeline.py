from __future__ import annotations

"""End-to-end ingestion pipeline."""

import logging
from pathlib import Path

from grounded_rag.ingestion.chunker import SectionAwareChunker
from grounded_rag.ingestion.loaders import DocumentLoader
from grounded_rag.models import Chunk, SourceDocument
from grounded_rag.utils.text import stable_hash

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, loader: DocumentLoader, chunker: SectionAwareChunker) -> None:
        self.loader = loader
        self.chunker = chunker

    def ingest_directory(self, source_dir: Path) -> tuple[list[SourceDocument], list[Chunk]]:
        documents = self.loader.load_directory(source_dir)
        unique_documents = self._deduplicate(documents)
        chunks: list[Chunk] = []
        for document in unique_documents:
            chunks.extend(self.chunker.chunk_document(document))
        logger.info(
            "Ingested %s unique documents into %s chunks.",
            len(unique_documents),
            len(chunks),
        )
        return unique_documents, chunks

    def _deduplicate(self, documents: list[SourceDocument]) -> list[SourceDocument]:
        seen_hashes: set[str] = set()
        unique_documents: list[SourceDocument] = []
        for document in documents:
            content_hash = stable_hash(
                "".join(section.text for section in document.sections) + document.title
            )
            if content_hash in seen_hashes:
                logger.info("Skipping duplicate document: %s", document.source_path)
                continue
            seen_hashes.add(content_hash)
            unique_documents.append(document)
        return unique_documents
