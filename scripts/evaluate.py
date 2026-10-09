"""Evaluation: 12 questions with expected answer keywords.

For each question we check whether ANY of the expected keywords appears in the
top-k retrieved chunks (retrieval hit) and whether it appears in the generated
answer (answer hit). Also records per-question latency. Writes JSON + markdown
report to artifacts/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

QUESTIONS = [
    {"q": "What are the three main paradigms of machine learning?",
     "keywords": ["supervised", "unsupervised", "reinforcement"]},
    {"q": "What is the difference between classification and regression?",
     "keywords": ["discrete", "continuous"]},
    {"q": "How does k-means clustering work?",
     "keywords": ["centroid"]},
    {"q": "What is backpropagation used for in neural networks?",
     "keywords": ["gradient", "chain rule"]},
    {"q": "What is overfitting and how can it be prevented?",
     "keywords": ["regularization", "cross-validation"]},
    {"q": "What does k-fold cross-validation do?",
     "keywords": ["folds", "averaged"]},
    {"q": "What happens if the learning rate is too large in gradient descent?",
     "keywords": ["diverge", "overshoot"]},
    {"q": "How does a random forest reduce overfitting compared to a single tree?",
     "keywords": ["bagging", "bootstrap", "variance"]},
    {"q": "What is TF-IDF?",
     "keywords": ["term frequency", "inverse document frequency"]},
    {"q": "What does a vector database like FAISS do?",
     "keywords": ["nearest", "similarity", "embeddings"]},
    {"q": "What are the five stages of a RAG pipeline?",
     "keywords": ["ingestion", "retrieval", "generation"]},
    {"q": "When is accuracy a misleading metric?",
     "keywords": ["imbalanced"]},
]


def evaluate(pipeline, top_k: int = 5) -> dict:
    rows = []
    for item in QUESTIONS:
        t0 = time.perf_counter()
        result = pipeline.ask(item["q"], top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        retrieved_text = " ".join(r["text"].lower() for r in result.retrieved)
        answer_text = result.answer.lower()
        kw_hits_retrieved = [k for k in item["keywords"] if k.lower() in retrieved_text]
        kw_hits_answer = [k for k in item["keywords"] if k.lower() in answer_text]
        rows.append({
            "question": item["q"],
            "keywords": item["keywords"],
            "retrieval_hit": len(kw_hits_retrieved) > 0,
            "answer_hit": len(kw_hits_answer) > 0,
            "keywords_in_retrieved": kw_hits_retrieved,
            "keywords_in_answer": kw_hits_answer,
            "latency_ms": round(latency_ms, 1),
        })
    n = len(rows)
    summary = {
        "n_questions": n,
        "top_k": top_k,
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in rows) / n, 4),
        "answer_hit_rate": round(sum(r["answer_hit"] for r in rows) / n, 4),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 1),
        "p95_latency_ms": round(sorted(r["latency_ms"] for r in rows)[int(n * 0.95) - 1], 1),
        "embedder": pipeline.embedder.name,
        "backend": pipeline.backend.name,
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    ap.add_argument("--index", default=str(ROOT / "artifacts" / "index"))
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline.load(args.index)
    report = evaluate(pipeline, top_k=args.top_k)

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2))

    s = report["summary"]
    lines = ["# RAG Evaluation Report", "",
             f"Questions: {s['n_questions']} | top_k={s['top_k']} | "
             f"embedder={s['embedder']} | backend={s['backend']}", "",
             f"- Retrieval hit-rate: **{s['retrieval_hit_rate']:.2%}**",
             f"- Answer hit-rate: **{s['answer_hit_rate']:.2%}**",
             f"- Mean latency: **{s['mean_latency_ms']} ms** "
             f"(p95: {s['p95_latency_ms']} ms)", "",
             "| # | Question | Retrieval hit | Answer hit | Latency (ms) |",
             "|---|----------|---------------|------------|--------------|"]
    for i, r in enumerate(report["rows"], 1):
        lines.append(f"| {i} | {r['question']} | "
                     f"{'yes' if r['retrieval_hit'] else 'NO'} | "
                     f"{'yes' if r['answer_hit'] else 'NO'} | {r['latency_ms']} |")
    (out_dir / "eval_report.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
