from __future__ import annotations

"""Persistence for chunks and dense vectors."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from grounded_rag.exceptions import IndexNotReadyError
from grounded_rag.models import Chunk


@dataclass(slots=True)
class PersistedIndex:
    chunks: list[Chunk]
    embeddings: list[list[float]]
    metadata: dict[str, Any]


class IndexRepository:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.artifact_dir / "chunks.jsonl"
        self.embeddings_path = self.artifact_dir / "dense_embeddings.json"
        self.manifest_path = self.artifact_dir / "manifest.json"

    def save(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        metadata: dict[str, Any],
    ) -> None:
        with self.chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict()) + "\n")

        dense_payload = [
            {"chunk_id": chunk.chunk_id, "vector": vector}
            for chunk, vector in zip(chunks, embeddings)
        ]
        self.embeddings_path.write_text(
            json.dumps(dense_payload, indent=2),
            encoding="utf-8",
        )
        self.manifest_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def load(self) -> PersistedIndex:
        if not self.chunks_path.exists() or not self.embeddings_path.exists():
            raise IndexNotReadyError(
                f"No persisted index found under {self.artifact_dir}. Run ingestion first."
            )

        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in self.chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        embedding_records = json.loads(self.embeddings_path.read_text(encoding="utf-8"))
        vector_by_chunk_id = {
            record["chunk_id"]: [float(value) for value in record["vector"]]
            for record in embedding_records
        }
        embeddings = [vector_by_chunk_id[chunk.chunk_id] for chunk in chunks]
        metadata = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.exists()
            else {}
        )
        return PersistedIndex(chunks=chunks, embeddings=embeddings, metadata=metadata)
