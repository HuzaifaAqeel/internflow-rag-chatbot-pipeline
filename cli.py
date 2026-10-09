#!/usr/bin/env python3
"""CLI: ask the RAG chatbot a question from the terminal.

Usage:
    python cli.py "What is overfitting?"
    python cli.py --top-k 3 "How does FAISS work?"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from src.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the RAG chatbot a question.")
    ap.add_argument("question", help="Question to ask about the knowledge base")
    ap.add_argument("--index", default=str(ROOT / "artifacts" / "index"))
    ap.add_argument("--model", default=os.environ.get("RAG_MODEL", "all-MiniLM-L6-v2"))
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    pipe = RAGPipeline.load(args.index, model_name=args.model)
    result = pipe.ask(args.question, top_k=args.top_k)

    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for i, c in enumerate(result.citations, 1):
        print(f"  [{i}] {c['title']} ({c['doc_id']}#chunk{c['chunk_index']}) "
              f"score={c['retrieval_score']}")
    print(f"\n(backend={result.backend}, embedder={result.embedder}, "
          f"latency={result.latency_ms} ms)")


if __name__ == "__main__":
    main()
