"""Simple semantic memory store using local embeddings."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from .embedder import get_embedder


@dataclass
class MemoryEntry:
    id: str
    created_at: str
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)

    def as_json(self) -> dict:
        payload = asdict(self)
        return payload


class SemanticMemory:
    def __init__(self, path: str | Path = "seed/memory/memory.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[MemoryEntry] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._entries.append(MemoryEntry(**data))
                except Exception:
                    continue

    @property
    def entries(self) -> List[MemoryEntry]:
        return list(self._entries)

    def add(
        self,
        title: str,
        content: str,
        *,
        tags: Iterable[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryEntry:
        embedder = get_embedder()
        embedding = embedder.embed([f"{title}\n{content}"])[0]
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            title=title,
            content=content,
            tags=list(tags or []),
            metadata=metadata or {},
            embedding=embedding,
        )
        self._entries.append(entry)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.as_json(), ensure_ascii=False) + "\n")
        return entry

    def query(self, text: str, *, top_k: int = 5) -> List[tuple[MemoryEntry, float]]:
        if not self._entries:
            return []
        embedder = get_embedder()
        query_vec = embedder.embed([text])[0]
        results = []
        for entry in self._entries:
            score = _cosine_similarity(query_vec, entry.embedding)
            results.append((entry, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
