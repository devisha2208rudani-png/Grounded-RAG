from __future__ import annotations

"""Configuration management for the Grounded RAG project."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in local tests
    def load_dotenv() -> bool:
        return False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


@dataclass(slots=True)
class Settings:
    source_dir: Path
    artifact_dir: Path
    cache_path: Path
    embedding_provider: str
    embedding_model: str
    llm_model: str
    llm_base_url: str | None
    llm_api_key: str | None
    top_k: int
    candidate_pool_size: int
    chunk_size: int
    chunk_overlap: int
    hybrid_alpha: float
    enable_rerank: bool
    confidence_threshold: float
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        artifact_dir = _env_path("RAG_ARTIFACT_DIR", "./.artifacts")
        cache_path = _env_path("RAG_CACHE_PATH", str(artifact_dir / "cache.sqlite3"))
        return cls(
            source_dir=_env_path("RAG_SOURCE_DIR", "./data/sample_docs"),
            artifact_dir=artifact_dir,
            cache_path=cache_path,
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "sentence_transformer"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "intfloat/e5-base-v2"),
            llm_model=os.getenv("RAG_LLM_MODEL", "gpt-4.1-mini"),
            llm_base_url=os.getenv("RAG_LLM_BASE_URL") or None,
            llm_api_key=os.getenv("RAG_LLM_API_KEY") or None,
            top_k=int(os.getenv("RAG_TOP_K", "4")),
            candidate_pool_size=int(os.getenv("RAG_CANDIDATE_POOL_SIZE", "8")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "180")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "40")),
            hybrid_alpha=float(os.getenv("RAG_HYBRID_ALPHA", "0.6")),
            enable_rerank=_env_bool("RAG_ENABLE_RERANK", True),
            confidence_threshold=float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.28")),
            log_level=os.getenv("RAG_LOG_LEVEL", "INFO"),
        )
