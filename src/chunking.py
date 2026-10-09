"""Chunking: split markdown documents into overlapping word-based chunks.

Strategy: split on blank lines into paragraphs, keep markdown headings attached
to the paragraph that follows them, then greedily pack paragraphs into chunks
of at most `chunk_size` words with `overlap` words of carry-over between
consecutive chunks. Deterministic and dependency-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    doc_id: str      # source file name, e.g. "01_what_is_ml.md"
    title: str       # first markdown heading in the doc
    chunk_index: int


def _paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Attach a lone heading line to the paragraph that follows it.
    merged: List[str] = []
    pending_heading = ""
    for p in parts:
        if re.match(r"^#{1,6}\s+\S", p) and "\n" not in p:
            pending_heading = (pending_heading + " " + p).strip() if pending_heading else p
            continue
        if pending_heading:
            p = pending_heading + "\n" + p
            pending_heading = ""
        merged.append(p)
    if pending_heading:
        merged.append(pending_heading)
    return merged


def chunk_text(text: str, doc_id: str, title: str, chunk_size: int = 220,
               overlap: int = 40) -> List[Chunk]:
    chunks: List[Chunk] = []
    current_words: List[str] = []

    def flush() -> None:
        if current_words:
            chunks.append(Chunk(
                text=" ".join(current_words),
                doc_id=doc_id,
                title=title,
                chunk_index=len(chunks),
            ))

    def push_words(words: List[str]) -> None:
        """Add words, splitting into chunk_size windows when a single
        paragraph is longer than the chunk size."""
        nonlocal current_words
        i = 0
        while i < len(words):
            room = chunk_size - len(current_words)
            if room <= 0:
                flush()
                current_words = current_words[-overlap:] if overlap else []
                room = chunk_size - len(current_words)
            take = min(room, len(words) - i)
            current_words.extend(words[i:i + take])
            i += take
            if len(current_words) >= chunk_size:
                flush()
                current_words = current_words[-overlap:] if overlap else []

    for para in _paragraphs(text):
        words = para.split()
        if len(current_words) + len(words) <= chunk_size or not current_words:
            if len(words) > chunk_size:
                flush()
                push_words(words)
            else:
                current_words.extend(words)
        else:
            flush()
            # carry overlap words into the next chunk
            current_words = current_words[-overlap:] + words if overlap else words
            if len(current_words) > chunk_size:
                tail = current_words
                current_words = []
                push_words(tail)
    flush()
    return chunks


def chunk_document(path, chunk_size: int = 220, overlap: int = 40) -> List[Chunk]:
    from pathlib import Path
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    title = m.group(1).strip() if m else path.stem
    return chunk_text(text, doc_id=path.name, title=title,
                      chunk_size=chunk_size, overlap=overlap)
