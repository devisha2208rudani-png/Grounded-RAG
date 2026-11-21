from __future__ import annotations

"""Document loading utilities."""

import logging
import re
from pathlib import Path

from grounded_rag.exceptions import IngestionError
from grounded_rag.models import DocumentSection, SourceDocument
from grounded_rag.utils.text import clean_text, stable_hash

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


class DocumentLoader:
    """Loads PDFs, Markdown, and text files into structured document objects."""

    def load_directory(self, source_dir: Path) -> list[SourceDocument]:
        if not source_dir.exists():
            raise IngestionError(f"Source directory does not exist: {source_dir}")

        documents: list[SourceDocument] = []
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            documents.append(self.load_file(path))
        return documents

    def load_file(self, path: Path) -> SourceDocument:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix in {".txt", ".md"}:
            return self._load_text_like(path)
        raise IngestionError(f"Unsupported document type: {path.suffix}")

    def _load_pdf(self, path: Path) -> SourceDocument:
        try:
            import fitz
        except ImportError as exc:
            raise IngestionError(
                "PyMuPDF is required to ingest PDF files. Install the project dependencies first."
            ) from exc

        logger.info("Loading PDF document: %s", path)
        document = fitz.open(path)
        sections: list[DocumentSection] = []

        for page_number, page in enumerate(document, start=1):
            page_text = clean_text(page.get_text("text"))
            if not page_text:
                continue
            sections.append(
                DocumentSection(
                    section_id=f"page-{page_number}",
                    text=page_text,
                    metadata={"page_number": page_number},
                )
            )

        if not sections:
            raise IngestionError(f"No text extracted from PDF: {path}")

        document_id = self._build_document_id(path, "\n".join(section.text for section in sections))
        return SourceDocument(
            document_id=document_id,
            title=path.stem,
            source_path=str(path),
            sections=sections,
            metadata={"file_type": "pdf", "page_count": len(sections)},
        )

    def _load_text_like(self, path: Path) -> SourceDocument:
        logger.info("Loading text document: %s", path)
        raw_text = clean_text(path.read_text(encoding="utf-8"))
        if not raw_text:
            raise IngestionError(f"Empty document: {path}")

        sections = self._split_sections(raw_text)
        document_id = self._build_document_id(path, raw_text)
        return SourceDocument(
            document_id=document_id,
            title=path.stem,
            source_path=str(path),
            sections=sections,
            metadata={"file_type": path.suffix.lstrip(".")},
        )

    def _split_sections(self, text: str) -> list[DocumentSection]:
        heading_pattern = re.compile(r"^(#+)\s+(.*)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))
        if not matches:
            return [DocumentSection(section_id="section-1", text=text, metadata={"section_title": "body"})]

        sections: list[DocumentSection] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = clean_text(text[start:end])
            if not body:
                continue
            title = clean_text(match.group(2))
            sections.append(
                DocumentSection(
                    section_id=f"section-{index + 1}",
                    text=body,
                    metadata={"section_title": title},
                )
            )
        return sections or [DocumentSection(section_id="section-1", text=text, metadata={"section_title": "body"})]

    def _build_document_id(self, path: Path, content: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
        return slug or stable_hash(f"{path.name}:{content}")[:16]
