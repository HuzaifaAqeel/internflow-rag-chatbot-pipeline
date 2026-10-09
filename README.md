# RAG Chatbot Pipeline with Vector DB

A working **Retrieval-Augmented Generation (RAG)** chatbot for an AI Engineer
internship task (Varolline — supervisor: Shahzaib Khan). It answers questions
about a local machine-learning knowledge base by retrieving the most relevant
document chunks from a **FAISS** vector store and generating a cited answer —
no API key required.

## Architecture

```
corpus/*.md ──▶ chunking ──▶ embeddings ──▶ FAISS index ──▶ retriever ──▶ generator
  (14 docs)     (220-word      (TF-IDF sparse      (IndexFlatIP,      (top-k cosine)   (extractive Q&A
                 overlapping     vectors,            cosine sim)                         + citations,
                 chunks)         L2-normalized,                                          or optional LLM)
                               or optional
                               dense model)
                                        ▲
                                        │  artifacts/index/  (persisted)
                                        ▼
   ┌──────────────┐   POST /ask   ┌─────────────────┐
   │  CLI (cli.py)│──────────────▶│  FastAPI (app/) │──▶ {answer, citations, scores, latency}
   └──────────────┘  GET /health └─────────────────┘      GET /stats
```

**Pipeline stages:** ingest → chunk → embed → index → retrieve → generate.
The generator is pluggable: the default **extractive backend** selects the
most query-relevant sentences from retrieved chunks (zero secrets, works
offline); setting `RAG_LLM_API_KEY` switches to an OpenAI-compatible LLM
backend that answers strictly from the retrieved context.

## Project layout

```
rag-chatbot-pipeline-vector-db/
├── corpus/                 # 14 sample ML docs (hand-written, see docs/CORPUS.md)
├── src/
│   ├── chunking.py         # paragraph-aware overlapping chunker
│   ├── embeddings.py       # SentenceTransformerEmbedder + TF-IDF fallback
│   ├── vectorstore.py      # FAISS IndexFlatIP wrapper (save/load)
│   ├── generator.py        # extractive + optional LLM answer backends
│   └── pipeline.py         # RAGPipeline: retrieve → generate
├── app/main.py             # FastAPI: GET /health, GET /stats, POST /ask
├── cli.py                  # terminal Q&A
├── scripts/
│   ├── build_index.py      # corpus → FAISS index (artifacts/index/)
│   └── evaluate.py         # 12-question eval: hit-rate + latency
├── tests/test_rag.py       # pytest suite (unit + API)
├── demo/sample_qa.md       # sample questions with answers
├── docs/CORPUS.md          # how the sample corpus was created
└── artifacts/              # generated: index, eval reports (git-ignored)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the index (TF-IDF embedder: fully offline, no downloads needed)
python scripts/build_index.py

# 2. Ask questions from the terminal
python cli.py "What is overfitting and how can it be prevented?"

# 3. Or serve the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API example:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does FAISS search vectors?", "top_k": 5}'
```

```json
{
  "question": "How does FAISS search vectors?",
  "answer": "FAISS (Facebook AI Similarity Search) is a library for efficient nearest-neighbour search over dense vectors...",
  "citations": [
    {"doc_id": "10_embeddings.md", "title": "Embeddings and Vector Search",
     "chunk_index": 0, "retrieval_score": 0.78}
  ],
  "latency_ms": 42.3,
  "backend": "extractive",
  "embedder": "tfidf-fallback"
}
```

Optional LLM answers (any OpenAI-compatible endpoint):

```bash
export RAG_LLM_API_KEY="sk-..."
export RAG_LLM_BASE_URL="https://api.openai.com/v1"   # or your provider
export RAG_LLM_MODEL="gpt-4o-mini"
python cli.py "What are the five stages of a RAG pipeline?"
```

## Evaluation

`python scripts/evaluate.py` runs 12 questions with known answer keywords over
the sample corpus and reports retrieval hit-rate (is an expected keyword in the
top-5 chunks?) and answer hit-rate (is it in the generated answer?), plus
latency. Full report: `artifacts/eval_report.md`.

<!-- EVAL_RESULTS -->
Measured with `python scripts/evaluate.py` (12 questions, top_k=5,
TF-IDF embedder, extractive backend, local machine):

| Metric | Value |
|---|---|
| Retrieval hit-rate | **83.33%** (10/12 questions have an expected keyword in the top-5 chunks) |
| Answer hit-rate | **83.33%** (10/12 answers contain an expected keyword) |
| Mean latency | **1.8 ms** per question (p95: 2.0 ms) |
| Test suite | **7/7 pytest tests pass** |
| API | `GET /health`, `GET /stats`, `POST /ask` verified live via uvicorn + curl |

The 2 misses are keyword-stringency artifacts (the retrieved chunks do discuss
the topic but phrase it without the exact expected keyword) — see
`artifacts/eval_report.md` for the per-question breakdown.

## Design decisions

- **FAISS `IndexFlatIP` on L2-normalized vectors** = exact cosine similarity.
  Exact search is the right call at this corpus scale; IVF/HNSW would be used
  past ~1M vectors.
- **Chunking:** 220-word chunks with 40-word overlap, paragraph-aware, headings
  kept with their section — avoids cutting explanations mid-sentence.
- **Embedder:** TF-IDF (1–2 grams, sublinear TF) is the default — it runs fully
  offline with zero downloads, which keeps the project self-contained. The
  code is pluggable: if `sentence-transformers` is installed, `default_embedder()`
  uses a dense model (e.g. `all-MiniLM-L6-v2`) instead, and `/stats` always
  reports which embedder is active so results stay honest.
- **Extractive default generator:** no API key, fully reproducible answers with
  verbatim citations — appropriate for a knowledge-base chatbot where
  faithfulness matters more than fluency.

## Limitations

1. The corpus is small, synthetic, and hand-written — retrieval metrics do not
   transfer to noisy real-world document collections.
2. The extractive backend quotes sentences; it cannot synthesize across chunks
   or answer questions needing multi-hop reasoning.
3. Ingestion supports markdown/text only — no PDF, HTML, or OCR pipeline.
4. No reranking stage (e.g. cross-encoder); top-k comes straight from vector
   similarity.
5. The FAISS index is rebuilt offline; there is no incremental add/delete API.

## Future improvements

1. Add a cross-encoder reranker on the top-20 candidates before generation, and
   hybrid (dense + BM25) retrieval for better keyword-heavy queries.
2. Incremental indexing with document versioning, plus a PDF/HTML ingestion
   path so the bot can serve real company documentation.
