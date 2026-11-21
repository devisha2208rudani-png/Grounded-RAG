from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grounded_rag.ingestion.chunker import ChunkingConfig, SectionAwareChunker
from grounded_rag.models import DocumentSection, SourceDocument
from grounded_rag.utils.text import tokenize


class ChunkingTests(unittest.TestCase):
    def test_section_aware_chunker_respects_overlap(self) -> None:
        document = SourceDocument(
            document_id="doc-1",
            title="Design Notes",
            source_path="design.md",
            sections=[
                DocumentSection(
                    section_id="section-1",
                    text=(
                        "Retrieval uses hybrid search. "
                        "Dense search captures semantic similarity. "
                        "BM25 improves exact phrase matching. "
                        "A reranker refines the final ordering."
                    ),
                    metadata={"section_title": "Retrieval"},
                )
            ],
        )

        chunker = SectionAwareChunker(ChunkingConfig(chunk_size=8, chunk_overlap=3))
        chunks = chunker.chunk_document(document)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["section_title"], "Retrieval")
        overlap_window = tokenize(chunks[0].text)[-4:]
        next_chunk_tokens = tokenize(chunks[1].text)
        self.assertTrue(any(token in next_chunk_tokens for token in overlap_window))


if __name__ == "__main__":
    unittest.main()
