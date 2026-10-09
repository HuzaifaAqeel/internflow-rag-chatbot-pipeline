"""FastAPI service: POST /ask (RAG question answering), GET /health, GET /stats."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", ROOT / "artifacts" / "index"))
MODEL_NAME = os.environ.get("RAG_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation over a local ML knowledge base (FAISS vector store).",
    version="1.0.0",
)

pipeline: RAGPipeline | None = None


@app.on_event("startup")
def _load() -> None:
    global pipeline
    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"Index not found at {INDEX_DIR}. Build it first: python scripts/build_index.py"
        )
    pipeline = RAGPipeline.load(INDEX_DIR, model_name=MODEL_NAME)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          description="Natural-language question about the knowledge base")
    top_k: int = Field(default=TOP_K, ge=1, le=10)


class Citation(BaseModel):
    doc_id: str
    title: str
    chunk_index: int
    retrieval_score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    latency_ms: float
    backend: str
    embedder: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_loaded": pipeline is not None}


@app.get("/stats")
def stats() -> dict:
    if pipeline is None:
        raise HTTPException(503, "index not loaded")
    return pipeline.stats()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if pipeline is None:
        raise HTTPException(503, "index not loaded")
    result = pipeline.ask(req.question, top_k=req.top_k)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
        latency_ms=result.latency_ms,
        backend=result.backend,
        embedder=result.embedder,
    )
