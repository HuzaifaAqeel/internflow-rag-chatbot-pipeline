#!/usr/bin/env python3
"""Demo: ask the RAG chatbot a set of sample questions and print Q&A + citations."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

from src.pipeline import RAGPipeline  # noqa: E402

SAMPLE_QUESTIONS = [
    "What are the three main paradigms of machine learning?",
    "What is overfitting and how can it be prevented?",
    "How does a random forest reduce overfitting?",
    "What does a vector database like FAISS do?",
    "What are the five stages of a RAG pipeline?",
    "When is accuracy a misleading metric?",
]


def main() -> None:
    pipe = RAGPipeline.load(ROOT / "artifacts" / "index")
    lines = ["# RAG Chatbot — Sample Q&A", ""]
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        res = pipe.ask(q)
        print(f"\n{'='*70}\nQ{i}: {q}\n\nA: {res.answer}\n")
        print("Sources:")
        for j, c in enumerate(res.citations, 1):
            print(f"  [{j}] {c['title']} ({c['doc_id']}#chunk{c['chunk_index']}) "
                  f"score={c['retrieval_score']}")
        print(f"(latency={res.latency_ms} ms, backend={res.backend}, "
              f"embedder={res.embedder})")
        lines += [f"## Q{i}: {q}", "", f"**A:** {res.answer}", "", "Sources:"]
        lines += [f"- {c['title']} (`{c['doc_id']}`#chunk{c['chunk_index']}, "
                  f"score={c['retrieval_score']})" for c in res.citations]
        lines += ["", f"_latency={res.latency_ms} ms, backend={res.backend}, "
                      f"embedder={res.embedder}_", ""]
    (ROOT / "demo" / "sample_qa.md").write_text("\n".join(lines))
    print(f"\n[demo] wrote demo/sample_qa.md")


if __name__ == "__main__":
    main()
