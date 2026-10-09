"""Pluggable text embedders: dense (sentence-transformers) with an offline TF-IDF fallback.

The pipeline prefers a real sentence-embedding model when available, and falls
back to TF-IDF vectors otherwise. Both produce L2-normalized float32 vectors so
cosine similarity == inner product in the FAISS index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np

try:  # optional dependency
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False


class Embedder(ABC):
    name: str = "base"

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized float32 vectors, shape (n_texts, dim)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _ST_AVAILABLE:
            raise RuntimeError("sentence-transformers is not installed")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


class TfidfEmbedder(Embedder):
    """Offline fallback: TF-IDF vectors, L2-normalized. Fit on the corpus."""

    name = "tfidf-fallback"

    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), sublinear_tf=True
        )
        self._fitted = False
        self._dim = max_features

    def fit(self, texts: List[str]) -> "TfidfEmbedder":
        self.vectorizer.fit(texts)
        self._fitted = True
        self._dim = len(self.vectorizer.vocabulary_)
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit() before encode()")
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    @property
    def dim(self) -> int:
        return self._dim

    # --- persistence helpers (vocabulary only; tiny) ---
    def save(self, path: Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocabulary": {k: int(v) for k, v in self.vectorizer.vocabulary_.items()},
            "idf": self.vectorizer.idf_.tolist(),
            "max_features": self.vectorizer.max_features,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "TfidfEmbedder":
        import json

        from sklearn.feature_extraction.text import TfidfVectorizer

        payload = json.loads(Path(path).read_text())
        obj = cls(max_features=payload.get("max_features", 4096))
        vec = TfidfVectorizer(
            max_features=payload.get("max_features", 4096),
            ngram_range=(1, 2),
            sublinear_tf=True,
            vocabulary=payload["vocabulary"],
        )
        # Fit on dummy data to initialise internals, then inject learned idf_.
        vec.fit(["dummy text for initialisation"])
        vec.idf_ = np.asarray(payload["idf"], dtype=np.float64)
        obj.vectorizer = vec
        obj._fitted = True
        obj._dim = len(payload["vocabulary"])
        return obj


def default_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Return a dense embedder if the model can be loaded, else TF-IDF fallback."""
    if _ST_AVAILABLE:
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:  # e.g. no cached model + no network
            print(f"[embeddings] dense model unavailable ({exc}); using TF-IDF fallback")
    else:
        print("[embeddings] sentence-transformers not installed; using TF-IDF fallback")
    return TfidfEmbedder()
