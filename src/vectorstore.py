"""FAISS-backed vector store: add/search/persist dense chunk vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from .chunking import Chunk


class VectorStore:
    """Exact inner-product (== cosine on normalized vectors) FAISS index."""

    def __init__(self, dim: int, embedder_name: str = ""):
        self.dim = dim
        self.embedder_name = embedder_name
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        assert vectors.shape[1] == self.dim, "vector dim mismatch"
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append((float(score), self.chunks[int(idx)]))
        return results

    def __len__(self) -> int:
        return self.index.ntotal

    # --- persistence ---
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        meta = [
            {"text": c.text, "doc_id": c.doc_id, "title": c.title,
             "chunk_index": c.chunk_index}
            for c in self.chunks
        ]
        (directory / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False))
        (directory / "store.json").write_text(json.dumps(
            {"dim": self.dim, "embedder_name": self.embedder_name}))

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        from .chunking import Chunk

        directory = Path(directory)
        info = json.loads((directory / "store.json").read_text())
        store = cls(dim=info["dim"], embedder_name=info.get("embedder_name", ""))
        store.index = faiss.read_index(str(directory / "faiss.index"))
        meta = json.loads((directory / "chunks.json").read_text())
        store.chunks = [Chunk(**m) for m in meta]
        return store
