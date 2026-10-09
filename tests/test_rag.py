"""Tests for the RAG chatbot pipeline (chunking, store, retrieval, API)."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAG_INDEX_DIR", str(ROOT / "artifacts" / "index"))

from src.chunking import chunk_text  # noqa: E402
from src.embeddings import TfidfEmbedder  # noqa: E402
from src.generator import ExtractiveBackend  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402


def test_chunking_basic():
    text = "# Title\n\n" + "word " * 500
    chunks = chunk_text(text, doc_id="d.md", title="Title", chunk_size=100, overlap=10)
    assert len(chunks) >= 4
    assert all(len(c.text.split()) <= 100 + 10 for c in chunks)
    assert chunks[0].doc_id == "d.md" and chunks[0].title == "Title"


def test_chunking_heading_attached():
    text = "# My Heading\n\nFirst paragraph here with enough words to stand alone."
    chunks = chunk_text(text, doc_id="d.md", title="My Heading",
                        chunk_size=220, overlap=0)
    assert len(chunks) == 1
    assert "My Heading" in chunks[0].text


def test_tfidf_embedder_normalized():
    emb = TfidfEmbedder().fit(["the cat sat", "the dog ran", "cats and dogs"])
    vecs = emb.encode(["the cat", "unknown words xyz"])
    assert vecs.shape == (2, emb.dim)
    import numpy as np
    norms = np.linalg.norm(vecs, axis=1)
    assert all(abs(n - 1.0) < 1e-5 or n == 0 for n in norms)


def test_vectorstore_roundtrip(tmp_path):
    import numpy as np
    from src.chunking import Chunk
    chunks = [Chunk(text=f"chunk {i}", doc_id="d.md", title="T", chunk_index=i)
              for i in range(4)]
    vecs = np.eye(4, dtype=np.float32)
    store = VectorStore(dim=4)
    store.add(vecs, chunks)
    assert len(store) == 4
    hits = store.search(np.array([0, 0, 1, 0], dtype=np.float32), top_k=2)
    assert hits[0][1].text == "chunk 2"
    assert hits[0][0] > hits[1][0]
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 4
    assert loaded.chunks[1].text == "chunk 1"


def test_extractive_backend_cites():
    from src.chunking import Chunk
    chunks = [(0.9, Chunk(text="Gradient descent minimizes loss by stepping "
                               "along the negative gradient direction.",
                          doc_id="a.md", title="GD", chunk_index=0))]
    ans, cites = ExtractiveBackend().generate("What is gradient descent?", chunks)
    assert "gradient" in ans.lower()
    assert cites and cites[0]["doc_id"] == "a.md"


def _pipeline():
    return RAGPipeline.load(ROOT / "artifacts" / "index")


def test_pipeline_ask_relevant():
    pipe = _pipeline()
    res = pipe.ask("What is overfitting?", top_k=3)
    assert res.answer and len(res.answer.split()) >= 5
    assert res.citations, "answer must carry citations"
    assert res.latency_ms > 0
    blob = (res.answer + " ".join(c["title"] for c in res.citations)).lower()
    assert "overfit" in blob


def test_api_endpoints():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200 and h.json()["status"] == "ok"
        s = client.get("/stats")
        assert s.status_code == 200 and s.json()["chunks_indexed"] > 0
        r = client.post("/ask", json={"question": "What is a vector database?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] and body["citations"]
        assert body["latency_ms"] > 0
        # validation: too-short question rejected
        bad = client.post("/ask", json={"question": "hi"})
        assert bad.status_code == 422
