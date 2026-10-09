"""Answer generation backends.

The default `ExtractiveBackend` needs no API key: it picks the most relevant
sentences from the retrieved chunks and presents them with citations.
An optional OpenAI-compatible LLM backend is used automatically when the
`RAG_LLM_API_KEY` environment variable is set (any OpenAI-compatible
`/chat/completions` endpoint via `RAG_LLM_BASE_URL`, default api.openai.com).
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

from .chunking import Chunk


_STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "and", "or", "to", "in", "how",
    "does", "do", "are", "why", "which", "for", "on", "with", "it", "its",
    "be", "been", "being", "can", "could", "should", "would", "will", "shall",
    "may", "might", "must", "have", "has", "had", "having", "i", "you", "he",
    "she", "we", "they", "them", "this", "that", "these", "those", "as",
    "at", "by", "from", "into", "about", "between", "through", "during",
    "there", "their", "then", "than", "so", "such", "no", "not", "only",
    "own", "same", "too", "very", "me", "my", "your", "our", "if", "else",
    "when", "where", "while", "who", "whom", "s", "t", "d", "ll", "re", "ve",
}


def _sentences(text: str) -> List[str]:
    # strip markdown heading markers so answers read cleanly
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 4]


def _score_sentence(sent: str, query_terms: set) -> float:
    words = set(re.findall(r"[a-z0-9]+", sent.lower()))
    overlap = len(words & query_terms)
    return overlap / max(1, len(words) ** 0.5)


class ExtractiveBackend:
    """Zero-secret answer synthesis: best sentences from retrieved chunks."""

    name = "extractive"

    def generate(self, question: str, retrieved: List[Tuple[float, Chunk]],
                 max_sentences: int = 3) -> Tuple[str, List[dict]]:
        query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        # drop stopwords so scoring focuses on content words
        query_terms -= _STOPWORDS

        scored = []
        for rank, (score, chunk) in enumerate(retrieved):
            for sent in _sentences(chunk.text):
                scored.append((_score_sentence(sent, query_terms), score, rank, sent, chunk))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        picked, seen = [], set()
        for s_score, r_score, rank, sent, chunk in scored:
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append((sent, chunk, r_score))
            if len(picked) >= max_sentences:
                break

        if not picked:
            # graceful fallback: quote the top chunk's opening
            top_score, top_chunk = retrieved[0][0], retrieved[0][1]
            picked = [(top_chunk.text[:400], top_chunk, top_score)]

        answer = " ".join(s for s, _, _ in picked)
        citations = [
            {"doc_id": c.doc_id, "title": c.title, "chunk_index": c.chunk_index,
             "retrieval_score": round(float(rs), 4)}
            for _, c, rs in picked
        ]
        return answer, citations


class OpenAICompatibleBackend:
    """Optional LLM backend (env: RAG_LLM_API_KEY, RAG_LLM_BASE_URL, RAG_LLM_MODEL)."""

    name = "llm"

    def __init__(self):
        self.api_key = os.environ["RAG_LLM_API_KEY"]
        self.base_url = os.environ.get("RAG_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini")

    def generate(self, question: str, retrieved: List[Tuple[float, Chunk]],
                 max_sentences: int = 3):
        import httpx

        context = "\n\n".join(
            f"[Source {i+1}: {c.title} ({c.doc_id}#chunk{c.chunk_index})]\n{c.text}"
            for i, (_, c) in enumerate(retrieved)
        )
        system = ("You answer questions using ONLY the provided sources. "
                  "Cite sources like [Source 1]. If the answer is not in the "
                  "sources, say so honestly.")
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "temperature": 0,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user",
                                "content": f"Sources:\n{context}\n\nQuestion: {question}"}]},
            timeout=60,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        citations = [
            {"doc_id": c.doc_id, "title": c.title, "chunk_index": c.chunk_index,
             "retrieval_score": round(float(s), 4)}
            for s, c in retrieved[:max_sentences]
        ]
        return answer, citations


def default_backend():
    if os.environ.get("RAG_LLM_API_KEY"):
        try:
            return OpenAICompatibleBackend()
        except Exception as exc:  # pragma: no cover
            print(f"[generator] LLM backend failed ({exc}); using extractive fallback")
    return ExtractiveBackend()
