from __future__ import annotations

"""Command-line interface for the Grounded RAG project."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from grounded_rag.config import Settings
from grounded_rag.evaluation.runner import evaluate_pipeline
from grounded_rag.pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grounded RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build or rebuild the retrieval index.")
    ingest_parser.add_argument("--source-dir", type=Path, default=None)

    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question.")
    ask_parser.add_argument("question", type=str)
    ask_parser.add_argument("--top-k", type=int, default=None)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run retrieval and grounding evaluation.")
    evaluate_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/retrieval_eval.jsonl"),
    )

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI server.")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()
    pipeline = RAGPipeline(settings=settings)
    try:
        if args.command == "ingest":
            manifest = pipeline.build_index(source_dir=args.source_dir)
            print(json.dumps(manifest, indent=2))
            return

        if args.command == "ask":
            answer = pipeline.ask(args.question, top_k=args.top_k)
            print(answer.answer)
            print("\nCitations:")
            for citation in answer.citations:
                print(
                    f"- {citation.citation_label} | {citation.title} | score={citation.score:.3f} | {citation.source_path}"
                )
            return

        if args.command == "evaluate":
            summary = evaluate_pipeline(pipeline, args.dataset)
            print(json.dumps(asdict(summary), indent=2))
            return

        if args.command == "serve":
            import uvicorn

            uvicorn.run("grounded_rag.api:app", host=args.host, port=args.port, reload=False)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
