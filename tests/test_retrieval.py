from __future__ import annotations

import sys
import shutil
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST_ARTIFACT_ROOT = ROOT / ".test_artifacts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grounded_rag.config import Settings
from grounded_rag.embeddings.deterministic import DeterministicHashEmbedder
from grounded_rag.generation.clients import RuleBasedGenerator
from grounded_rag.pipeline import RAGPipeline


def build_settings(artifact_dir: Path) -> Settings:
    return Settings(
        source_dir=(ROOT / "data" / "sample_docs").resolve(),
        artifact_dir=artifact_dir,
        cache_path=artifact_dir / "cache.sqlite3",
        embedding_provider="deterministic",
        embedding_model="deterministic-hash",
        llm_model="offline",
        llm_base_url=None,
        llm_api_key=None,
        top_k=4,
        candidate_pool_size=8,
        chunk_size=80,
        chunk_overlap=20,
        hybrid_alpha=0.55,
        enable_rerank=True,
        confidence_threshold=0.25,
        log_level="INFO",
    )


class RetrievalTests(unittest.TestCase):
    def test_hybrid_retrieval_prefers_operations_doc_for_fallback_question(self) -> None:
        TEST_ARTIFACT_ROOT.mkdir(exist_ok=True)
        case_dir = TEST_ARTIFACT_ROOT / "retrieval_case"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        pipeline = RAGPipeline(
            settings=build_settings(case_dir),
            embedder=DeterministicHashEmbedder(),
            answer_generator=RuleBasedGenerator(),
        )
        pipeline.build_index()

        retrieval = pipeline.retrieve("What happens when retrieval confidence is low?")

        self.assertTrue(retrieval.results)
        self.assertEqual(retrieval.results[0].chunk.document_id, "operations")
        pipeline.close()
        shutil.rmtree(case_dir)


if __name__ == "__main__":
    unittest.main()
