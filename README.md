# Grounded RAG

Grounded RAG is a modular, local-first Retrieval-Augmented Generation system built from an exploratory notebook and refactored into a production-style Python package.

## Highlights

- Multi-format ingestion for PDF, Markdown, and plain text documents
- Section-aware chunking with configurable chunk size and overlap
- Hybrid retrieval with dense embeddings plus BM25 lexical search
- Metadata-aware reranking to improve precision on section-heavy queries
- Grounded answer generation with inline citation labels and source tracing
- Low-confidence fallback when retrieval evidence is weak
- SQLite caching for embeddings and repeated responses
- Persisted index artifacts for fast startup and reproducible demos
- Built-in evaluation for recall@k, MRR, citation hit rate, and term coverage
- CLI and FastAPI interfaces for local usage and service deployment


## Repository Layout

```text
.
|-- .github/workflows/ci.yml
|-- notebooks/
|   |-- llm_work_2.ipynb
|   `-- README.md
|-- data/
|   |-- sample_docs/
|   `-- eval/
|-- src/grounded_rag/
|   |-- ingestion/
|   |-- embeddings/
|   |-- index/
|   |-- retrieval/
|   |-- generation/
|   |-- evaluation/
|   |-- api.py
|   |-- cli.py
|   |-- config.py
|   |-- models.py
|   `-- pipeline.py
|-- tests/
|-- README.md
|-- CONTRIBUTING.md
|-- LICENSE
`-- pyproject.toml
```

## Quickstart

1. Create a virtual environment.

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the project.

   ```bash
   pip install -e .
   ```

3. Copy the environment template.

   ```bash
   copy .env.example .env
   ```

4. Build the index.

   ```bash
   grounded-rag ingest --source-dir data/sample_docs
   ```

5. Ask a grounded question.

   ```bash
   grounded-rag ask "What happens when retrieval confidence is low?"
   ```

## Configuration

Key environment variables are defined in [.env.example](./.env.example).

- `RAG_SOURCE_DIR`: directory to ingest
- `RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP`: chunking behavior
- `RAG_TOP_K` and `RAG_CANDIDATE_POOL_SIZE`: retrieval breadth
- `RAG_HYBRID_ALPHA`: weight between dense and lexical retrieval
- `RAG_ENABLE_RERANK`: toggles reranking
- `RAG_CONFIDENCE_THRESHOLD`: fallback trigger
- `RAG_LLM_*`: OpenAI-compatible generation backend

## Usage

Build or rebuild the retrieval index:

```bash
grounded-rag ingest --source-dir data/sample_docs
```

Run a question-answering query:

```bash
grounded-rag ask "Which evaluation metrics are tracked by the project?"
```

Run evaluation:

```bash
grounded-rag evaluate --dataset data/eval/retrieval_eval.jsonl
```

Start the API:

```bash
grounded-rag serve --host 127.0.0.1 --port 8000
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"How does the system combine dense and lexical retrieval?\"}"
```

## Evaluation

The project includes a small evaluation set in [retrieval_eval.jsonl](./data/eval/retrieval_eval.jsonl) and computes:

- `recall@k`
- `MRR`
- `citation_hit_rate`
- `expected_term_coverage`

This makes the project easier to discuss in interviews because retrieval quality and grounded answer quality are measured explicitly instead of judged only by manual spot checks.

## Development

Run tests locally:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same test suite on pushes and pull requests via [.github/workflows/ci.yml](./.github/workflows/ci.yml).

## Notes

- The original notebook prototype now lives in [notebooks/llm_work_2.ipynb](./notebooks/llm_work_2.ipynb).
- The maintained implementation lives in [src/grounded_rag](./src/grounded_rag).
- The dense index is local and portable by design, but the retrieval layer is modular enough to swap in FAISS or a hosted vector database later.
