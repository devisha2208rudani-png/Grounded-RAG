# System Architecture

The Grounded RAG system is organized as a modular Python package instead of a notebook-first workflow.
Document ingestion, chunking, indexing, retrieval, generation, and evaluation are separated into dedicated modules so each step can be tested and extended independently.

## Ingestion Pipeline

The ingestion stage supports PDF, Markdown, and plain text inputs.
PDF files are parsed page by page with PyMuPDF so the pipeline can preserve page-level provenance.
Markdown files are split by section headings, while text files are treated as a single logical body when no headings are available.

## Chunking Strategy

Chunking is section-aware and sentence-aware.
The chunker groups sentences until it reaches a configurable token budget and keeps a configurable overlap window between adjacent chunks.
This design reduces context fragmentation while keeping each chunk small enough for embedding and retrieval.

## Retrieval Strategy

Retrieval is hybrid rather than purely dense.
Dense similarity search uses sentence-transformer embeddings, while a BM25 retriever captures exact term matches and metadata-heavy queries.
The final ranking uses weighted score fusion and reciprocal-rank style boosting before an optional reranking pass.

## Grounded Answering

The generator receives only retrieved chunks and is instructed to answer with inline citations.
Each chunk has a citation label such as architecture:2 so the response can reference the exact supporting source.
If no external LLM is configured, the system falls back to an extractive summary instead of failing the request.

