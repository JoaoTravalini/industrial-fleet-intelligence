"""Deterministic lexical retrieval for trusted copilot knowledge chunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    """Trusted project knowledge used as supporting context."""

    id: str
    title: str
    content: str

    def to_source(self) -> dict[str, str]:
        return {"type": "knowledge", "id": self.id, "label": self.title}


def default_knowledge_path() -> Path:
    return Path(__file__).with_name("knowledge_base.json")


def load_knowledge(path: Path | None = None) -> tuple[KnowledgeChunk, ...]:
    raw = json.loads((path or default_knowledge_path()).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Knowledge base must be a JSON array.")
    chunks: list[KnowledgeChunk] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Knowledge chunk must be an object.")
        chunk = KnowledgeChunk(
            id=validate_text(item.get("id"), "id"),
            title=validate_text(item.get("title"), "title"),
            content=validate_text(item.get("content"), "content"),
        )
        chunks.append(chunk)
    ids = [chunk.id for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Knowledge chunk ids must be unique.")
    return tuple(chunks)


def validate_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Knowledge chunk {field_name} must be non-empty text.")
    return value.strip()


def tokenize(value: str) -> tuple[str, ...]:
    return tuple(token for token in TOKEN_PATTERN.findall(value.lower()) if token not in STOPWORDS)


def retrieve_knowledge(
    query: str,
    chunks: tuple[KnowledgeChunk, ...],
    *,
    top_k: int,
) -> tuple[KnowledgeChunk, ...]:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return ()
    scored: list[tuple[float, str, KnowledgeChunk]] = []
    for chunk in chunks:
        title_tokens = set(tokenize(chunk.title))
        content_tokens = set(tokenize(chunk.content))
        overlap = query_tokens & (title_tokens | content_tokens)
        if not overlap:
            continue
        title_boost = len(query_tokens & title_tokens) * 1.5
        score = len(overlap) + title_boost
        scored.append((score, chunk.id, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in scored[:top_k])
