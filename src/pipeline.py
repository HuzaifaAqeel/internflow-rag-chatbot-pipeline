"""End-to-end RAG pipeline: retrieve top-k chunks, generate a cited answer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .chunking import Chunk
from .embeddings import Embedder, TfidfEmbedder, default_embedder
from .generator import default_backend
from .vectorstore import VectorStore

INDEX_DIRNAME = "index"


@dataclass
class AskResult:
    question: str
    answer: str
    citations: List[dict]
    retrieved: List[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    backend: str = ""
    embedder: str = ""


class RAGPipeline:
    def __init__(self, store: VectorStore, embedder: Embedder, backend=None):
        self.store = store
        self.embedder = embedder
        self.backend = backend or default_backend()

    @classmethod
    def load(cls, index_dir: Path, model_name: str = "all-MiniLM-L6-v2",
             backend=None) -> "RAGPipeline":
        index_dir = Path(index_dir)
        store = VectorStore.load(index_dir)
        embedder: Embedder
        if store.embedder_name == TfidfEmbedder.name:
            embedder = TfidfEmbedder.load(index_dir / "tfidf.json")
        else:
            embedder = default_embedder(model_name)
            if isinstance(embedder, TfidfEmbedder):
                raise RuntimeError(
                    "Index was built with dense embeddings but the dense model "
                    "is unavailable now — rebuild the index or restore the model."
                )
        return cls(store, embedder, backend=backend)

    def ask(self, question: str, top_k: int = 5) -> AskResult:
        t0 = time.perf_counter()
        qvec = self.embedder.encode([question])[0]
        hits = self.store.search(qvec, top_k=top_k)
        answer, citations = self.backend.generate(question, hits)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        retrieved = [
            {"doc_id": c.doc_id, "title": c.title, "chunk_index": c.chunk_index,
             "score": round(float(s), 4), "text": c.text[:500]}
            for s, c in hits
        ]
        return AskResult(
            question=question, answer=answer, citations=citations,
            retrieved=retrieved, latency_ms=round(latency_ms, 1),
            backend=self.backend.name, embedder=self.embedder.name,
        )

    def stats(self) -> dict:
        return {
            "chunks_indexed": len(self.store),
            "embedding_dim": self.store.dim,
            "embedder": self.embedder.name,
            "backend": self.backend.name,
        }
