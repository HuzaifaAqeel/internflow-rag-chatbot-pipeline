"""Build the FAISS index from the corpus: chunk -> embed -> index -> persist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.chunking import chunk_document
from src.embeddings import TfidfEmbedder, default_embedder
from src.vectorstore import VectorStore


def build(corpus_dir: Path, index_dir: Path, model_name: str,
          chunk_size: int = 220, overlap: int = 40) -> dict:
    corpus_dir, index_dir = Path(corpus_dir), Path(index_dir)
    docs = sorted(corpus_dir.glob("*.md"))
    if not docs:
        raise SystemExit(f"No markdown documents found in {corpus_dir}")

    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    print(f"[build] {len(docs)} documents -> {len(chunks)} chunks")

    embedder = default_embedder(model_name)
    print(f"[build] embedder: {embedder.name}")
    if isinstance(embedder, TfidfEmbedder):
        embedder.fit([c.text for c in chunks])
    vectors = embedder.encode([c.text for c in chunks])
    print(f"[build] vectors: {vectors.shape}")

    store = VectorStore(dim=vectors.shape[1], embedder_name=embedder.name)
    store.add(vectors, chunks)
    store.save(index_dir)
    if isinstance(embedder, TfidfEmbedder):
        embedder.save(index_dir / "tfidf.json")

    summary = {
        "documents": len(docs),
        "chunks": len(chunks),
        "embedder": embedder.name,
        "dim": int(vectors.shape[1]),
        "index_dir": str(index_dir),
    }
    (index_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[build] index saved to {index_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the RAG vector index from the corpus.")
    ap.add_argument("--corpus", default=str(ROOT / "corpus"))
    ap.add_argument("--index", default=str(ROOT / "artifacts" / "index"))
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--chunk-size", type=int, default=220)
    ap.add_argument("--overlap", type=int, default=40)
    args = ap.parse_args()
    summary = build(args.corpus, args.index, args.model,
                    chunk_size=args.chunk_size, overlap=args.overlap)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
