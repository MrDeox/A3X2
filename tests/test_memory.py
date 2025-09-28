from pathlib import Path
from typing import Iterable, List

from a3x.memory.embedder import EmbeddingModel, set_embedder
from a3x.memory.store import SemanticMemory
from a3x.memory.insights import build_insight_payload
from a3x.autoeval import RunEvaluation, EvaluationSeed


class DummyEmbedder(EmbeddingModel):
    def __init__(self) -> None:  # type: ignore[override]
        pass

    def embed(self, texts: Iterable[str]) -> List[List[float]]:  # type: ignore[override]
        vectors: List[List[float]] = []
        for text in texts:
            total = sum(ord(ch) for ch in text)
            length = max(len(text), 1)
            vectors.append([total / 1000.0, length / 100.0])
        return vectors


def test_memory_store(tmp_path: Path) -> None:
    set_embedder(DummyEmbedder())
    memory_file = tmp_path / "memory.jsonl"
    store = SemanticMemory(memory_file)
    store.add("Run A", "content", tags=["tag"], metadata={"goal": "A"})
    store.add("Run B", "another", tags=["tag"], metadata={"goal": "B"})

    results = store.query("another", top_k=2)
    assert any(entry.metadata.get("goal") == "B" for entry, _ in results)
    set_embedder(None)


def test_insight_payload() -> None:
    evaluation = RunEvaluation(
        goal="demo",
        completed=True,
        iterations=3,
        failures=0,
        duration_seconds=0.5,
        timestamp="2025-01-01T00:00:00+00:00",
        seeds=[EvaluationSeed(description="Seed desc", seed_type="analysis")],
        metrics={"actions_success_rate": 1.0},
        capabilities=["core.diffing"],
    )

    title, content, tags, metadata = build_insight_payload(
        evaluation, {"core.diffing": {"success_rate": 0.99}}
    )
    assert "demo" in title
    assert "actions_success_rate" in content
    assert "core.diffing" in tags
    assert metadata["goal"] == "demo"

    set_embedder(None)
