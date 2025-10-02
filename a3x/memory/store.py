"""Simple semantic memory store using local embeddings."""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from ..constants import MEMORY_TOP_K_DEFAULT, MEMORY_TTL_DAYS
from .embedder import get_embedder
from ..cache import memory_cache_manager


@dataclass
class MemoryEntry:
    id: str
    created_at: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


class SemanticMemory:
    def __init__(self, path: str | Path = "seed/memory/memory.jsonl") -> None:
        self.path: Path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock: Lock = Lock()
        self._entries: list[MemoryEntry] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data: dict[str, Any] = json.loads(line)
                    self._entries.append(MemoryEntry(**data))
                except Exception:
                    continue
        with self._lock:
            self._prune_old_entries_locked()
            self._save_atomic()

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def add(
        self,
        title: str,
        content: str,
        *,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        embedder = get_embedder()
        text_to_embed = f"{title}\n{content}"

        # Try to get cached embedding first
        memory_cache = memory_cache_manager.get_cache("semantic_memory")
        cached_embedding = memory_cache.get_cached_embedding(
            text_to_embed,
            getattr(embedder, 'model_name', 'default'),
            getattr(embedder, 'method', 'sentence_transformer')
        )

        if cached_embedding is not None:
            embedding = cached_embedding
        else:
            embedding = embedder.embed([text_to_embed])[0]
            # Cache the embedding for future use
            memory_cache.cache_embedding(
                text_to_embed,
                embedding,
                getattr(embedder, 'model_name', 'default'),
                getattr(embedder, 'method', 'sentence_transformer')
            )
        entry: MemoryEntry = MemoryEntry(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            title=title,
            content=content,
            tags=list(tags or []),
            metadata=metadata or {},
            embedding=embedding,
        )
        with self._lock:
            self._entries.append(entry)
            self._prune_old_entries_locked()
            self._save_atomic()
        return entry

    def prune_old_entries(self) -> None:
        """Delete entries older than configured days in-memory only."""
        with self._lock:
            self._prune_old_entries_locked()

    def _save_atomic(self) -> None:
        """Atomically save all entries to JSONL file using tempfile.

        Callers must hold ``self._lock`` while invoking this method to
        guarantee consistency between the in-memory list and the file
        contents.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl.tmp",
            delete=False,
        ) as tmp:
            for entry in self._entries:
                tmp.write(json.dumps(entry.as_json(), ensure_ascii=False) + "\n")
            tmp_path: str = tmp.name
        os.replace(tmp_path, self.path)

    def _prune_old_entries_locked(self) -> None:
        now: datetime = datetime.now(timezone.utc)
        ttl: timedelta = timedelta(days=MEMORY_TTL_DAYS)
        cutoff: datetime = now - ttl
        self._entries = [
            entry
            for entry in self._entries
            if datetime.fromisoformat(entry.created_at) > cutoff
        ]

    def query(self, text: str, *, top_k: int = MEMORY_TOP_K_DEFAULT) -> list[tuple[MemoryEntry, float]]:
        if not self._entries:
            return []

        embedder = get_embedder()

        # Try to get cached query results first
        memory_cache = memory_cache_manager.get_cache("semantic_memory")
        cached_results = memory_cache.get_cached_query(text, top_k, None)

        if cached_results is not None:
            return cached_results

        # Compute query embedding (with caching)
        query_vec = embedder.embed([text])[0]

        results: list[tuple[MemoryEntry, float]] = []
        with self._lock:
            for entry in self._entries:
                # Use cached similarity calculation if available
                cached_similarity = memory_cache.get_cached_similarity(query_vec, entry.embedding)
                if cached_similarity is not None:
                    score = cached_similarity
                else:
                    score = _cosine_similarity(query_vec, entry.embedding)
                    # Cache the similarity calculation
                    memory_cache.cache_similarity(query_vec, entry.embedding, score)

                results.append((entry, score))

        results.sort(key=lambda item: item[1], reverse=True)
        query_results = results[:top_k]

        # Cache the query results
        memory_cache.cache_query(text, query_results, top_k, None)

        return query_results


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot: float = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a: float = math.sqrt(sum(a * a for a in vec_a))
    norm_b: float = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
