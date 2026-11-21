# Operations And Reliability

The project is designed to behave like a small production service rather than a one-off demo.
It includes structured logging, persistent caching, error handling, and index persistence so repeated runs do not recompute every step.

## Caching

Embedding vectors are cached in SQLite with a stable hash of the model name and source text.
Generated answers are also cached by normalized question, retrieval settings, and index version so repeated queries return quickly.

## Low Confidence Fallback

The retriever computes a confidence score from the fused ranking distribution.
If the confidence score is below the configured threshold, the system does not force a speculative answer.
Instead it returns a grounded fallback message that explains there is not enough evidence in the indexed documents.

## Error Handling

The code raises explicit exceptions for ingestion, configuration, index readiness, and generation failures.
This makes it easier to debug operational issues and keeps the API behavior predictable.

## Persistence

The build step writes chunk metadata, dense vectors, and a manifest file into the artifacts directory.
The service can reload the persisted index on startup without repeating document ingestion.

