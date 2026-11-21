from __future__ import annotations

"""FastAPI surface for the Grounded RAG system."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from grounded_rag.config import Settings
from grounded_rag.pipeline import RAGPipeline

app = FastAPI(title="Grounded RAG API", version="0.1.0")


class IngestRequest(BaseModel):
    source_dir: str | None = Field(default=None, description="Directory containing documents to index.")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.on_event("startup")
def startup() -> None:
    settings = Settings.from_env()
    app.state.pipeline = RAGPipeline(settings=settings)
    app.state.pipeline.ensure_index()


@app.on_event("shutdown")
def shutdown() -> None:
    app.state.pipeline.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, object]:
    pipeline: RAGPipeline = app.state.pipeline
    source_dir = Path(request.source_dir).resolve() if request.source_dir else None
    manifest = pipeline.build_index(source_dir=source_dir)
    return {"status": "indexed", "manifest": manifest}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    pipeline: RAGPipeline = app.state.pipeline
    try:
        answer = pipeline.ask(request.question, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "question": answer.question,
        "answer": answer.answer,
        "confidence": answer.confidence,
        "fallback_used": answer.fallback_used,
        "citations": [
            {
                "citation_label": citation.citation_label,
                "title": citation.title,
                "source_path": citation.source_path,
                "snippet": citation.snippet,
                "score": citation.score,
                "metadata": citation.metadata,
            }
            for citation in answer.citations
        ],
    }
