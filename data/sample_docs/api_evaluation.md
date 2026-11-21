# API Surface And Evaluation

The project exposes both a command-line interface and a FastAPI service.
The CLI supports ingest, ask, evaluate, and serve commands so the same pipeline can be used locally, in tests, or behind an API.

## API Endpoints

The API exposes a health endpoint for readiness checks, an ingest endpoint for rebuilding the index, and an ask endpoint for grounded question answering.
Each answer includes confidence, fallback status, and source citations to support downstream UI rendering.

## Evaluation

Evaluation is built into the project instead of being an afterthought.
The evaluation runner loads JSONL examples and computes recall at k, mean reciprocal rank, citation hit rate, and expected-term coverage.
This helps distinguish strong retrieval from good-sounding but weakly grounded generation.

## Interview Framing

The system is intentionally practical.
It is complex enough to demonstrate retrieval engineering, observability, and software design, but still small enough to explain clearly in an interview.
