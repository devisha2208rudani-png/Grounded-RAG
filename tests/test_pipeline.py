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


def build_settings(artifact_dir: Path, confidence_threshold: float) -> Settings:
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
        confidence_threshold=confidence_threshold,
        log_level="INFO",
    )


class PipelineTests(unittest.TestCase):
    def test_pipeline_returns_citations_for_supported_question(self) -> None:
        TEST_ARTIFACT_ROOT.mkdir(exist_ok=True)
        case_dir = TEST_ARTIFACT_ROOT / "pipeline_supported_case"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        pipeline = RAGPipeline(
            settings=build_settings(case_dir, confidence_threshold=0.2),
            embedder=DeterministicHashEmbedder(),
            answer_generator=RuleBasedGenerator(),
        )
        pipeline.build_index()

        answer = pipeline.ask("Which evaluation metrics are tracked by the project?")

        self.assertFalse(answer.fallback_used)
        self.assertGreaterEqual(len(answer.citations), 1)
        self.assertIn("Evidence:", answer.answer)
        pipeline.close()
        shutil.rmtree(case_dir)

    def test_pipeline_uses_fallback_when_confidence_is_low(self) -> None:
        TEST_ARTIFACT_ROOT.mkdir(exist_ok=True)
        case_dir = TEST_ARTIFACT_ROOT / "pipeline_fallback_case"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        pipeline = RAGPipeline(
            settings=build_settings(case_dir, confidence_threshold=0.55),
            embedder=DeterministicHashEmbedder(),
            answer_generator=RuleBasedGenerator(),
        )
        pipeline.build_index()

        answer = pipeline.ask("What is the corporate travel reimbursement policy?")

        self.assertTrue(answer.fallback_used)
        self.assertIn("enough grounded evidence", answer.answer.lower())
        pipeline.close()
        shutil.rmtree(case_dir)


if __name__ == "__main__":
    unittest.main()
