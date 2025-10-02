"""Utilities to convert evaluations into semantic memory entries."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from typing import TYPE_CHECKING

from ..actions import AgentState
from .store import MemoryEntry, SemanticMemory

if TYPE_CHECKING:  # pragma: no cover
    from ..agent import AgentResult
    from ..autoeval import RunEvaluation
    from ..planning import GoalPlan


def build_insight_payload(
    evaluation: RunEvaluation,
    capability_metrics: dict[str, dict[str, float | int | None]] | None = None,
    snapshot: str | None = None,
) -> tuple[str, str, list[str], dict]:
    status = "✅" if evaluation.completed else "⚠️"
    title = f"Run {evaluation.goal} {status}"
    lines = [
        f"goal: {evaluation.goal}",
        f"status: {'completed' if evaluation.completed else 'incomplete'}",
        f"iterations: {evaluation.iterations}",
        f"failures: {evaluation.failures}",
    ]
    if evaluation.metrics:
        metric_parts = [
            f"{name}={value:.3f}" for name, value in sorted(evaluation.metrics.items())
        ]
        lines.append("metrics: " + ", ".join(metric_parts))
    if evaluation.capabilities:
        lines.append("capabilities: " + ", ".join(sorted(evaluation.capabilities)))
    if evaluation.seeds:
        seeds_desc = "; ".join(seed.description for seed in evaluation.seeds)
        lines.append("seeds: " + seeds_desc)
    if capability_metrics and evaluation.capabilities:
        cap_lines: list[str] = []
        for cap in evaluation.capabilities:
            data = capability_metrics.get(cap)
            if not data:
                continue
            metrics_str = ", ".join(
                f"{k}={v}" for k, v in data.items() if isinstance(v, (int, float))
            )
            if metrics_str:
                cap_lines.append(f"{cap}: {metrics_str}")
        if cap_lines:
            lines.append("capability_metrics: " + " | ".join(cap_lines))
    content = "\n".join(lines)
    tags = sorted({*evaluation.capabilities, status})
    metadata = {
        "goal": evaluation.goal,
        "completed": evaluation.completed,
        "capabilities": evaluation.capabilities,
        "timestamp": evaluation.timestamp,
    }
    if snapshot:
        metadata["snapshot_hash"] = md5(snapshot.encode()).hexdigest()
    return title, content, tags, metadata


@dataclass
class Insight:
    """Simplified insight from memory entry for retrieval."""
    title: str
    content: str
    tags: list[str]
    metadata: dict
    created_at: str
    similarity: float = 0.0

    @classmethod
    def from_entry(cls, entry: MemoryEntry, similarity: float) -> Insight:
        return cls(
            title=entry.title,
            content=entry.content,
            tags=entry.tags,
            metadata=entry.metadata,
            created_at=entry.created_at,
            similarity=similarity,
        )


class StatefulRetriever:
    """Retriever for session continuity with dynamic state and derivation detection."""

    def __init__(self):
        self.store = SemanticMemory()
        self.last_snapshot_hash: str | None = None
        self.derivation_threshold = 0.1  # Simple diff ratio threshold for flag
        self.similarity_threshold = 0.5  # Lowered for better retrieval in structured content

    def retrieve_session_context(self, state: AgentState) -> list[Insight]:
        """Retrieve recent relevant insights based on state, filtered by similarity >0.7."""
        # Derive query from state
        history_summary = state.history_snapshot[:500] if state.history_snapshot else ""  # Truncate for query
        query_text = f"goal: {state.goal} actions_success_rate low metrics failures capabilities planning session context recent history: {history_summary} events"

        results = self.store.query(query_text, top_k=10)

        # Filter by similarity > self.similarity_threshold and sort by recency
        filtered = [(entry, score) for entry, score in results if score > self.similarity_threshold]
        recent = sorted(filtered, key=lambda x: x[0].created_at, reverse=True)[:5]

        insights = [Insight.from_entry(entry, score) for entry, score in recent]

        # Detect derivation on retrieved insights
        for insight in insights:
            if self._detect_derivation(state.history_snapshot, insight):
                insight.metadata["derivation_flagged"] = True

        return insights

    def _detect_derivation(self, current_snapshot: str, insight: Insight) -> bool:
        """Basic derivation detection: hash comparison with threshold on content changes."""
        if not current_snapshot or "snapshot_hash" not in insight.metadata:
            return False

        current_hash = md5(current_snapshot.encode()).hexdigest()
        stored_hash = insight.metadata.get("snapshot_hash", "")

        # Simple flag if hashes differ (threshold on hash diff not directly, but flag change)
        if current_hash != stored_hash:
            # Basic change flag; could compute Levenshtein but minimal
            insight.metadata["snapshot_hash"] = current_hash
            return True  # Flagged as derived/changed
        return False

    def update_snapshot_hash(self, snapshot: str) -> None:
        """Update last known snapshot hash for future comparisons."""
        self.last_snapshot_hash = md5(snapshot.encode()).hexdigest()


@dataclass
class RetrospectiveReport:
    """Synthetic retrospective persisted after each run."""

    goal: str
    completed: bool
    iterations: int
    failures: int
    duration_seconds: float | None
    metrics: dict[str, float]
    recommendations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "completed": self.completed,
            "iterations": self.iterations,
            "failures": self.failures,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RetrospectiveReport:
        return cls(
            goal=str(data.get("goal", "")),
            completed=bool(data.get("completed", False)),
            iterations=int(data.get("iterations", 0)),
            failures=int(data.get("failures", 0)),
            duration_seconds=(
                float(data.get("duration_seconds"))
                if data.get("duration_seconds") is not None
                else None
            ),
            metrics={
                str(k): float(v)
                for k, v in (data.get("metrics") or {}).items()
                if isinstance(v, (int, float))
            },
            recommendations=list(data.get("recommendations", []) or []),
            notes=list(data.get("notes", []) or []),
            created_at=str(data.get("created_at", datetime.now(timezone.utc).isoformat())),
        )


def build_retrospective(
    result: AgentResult,
    plan: GoalPlan | None,
    metrics: dict[str, float],
    *,
    alerts: Iterable[str] | None = None,
) -> RetrospectiveReport:
    recommendations: list[str] = []
    notes: list[str] = []
    alerts_list = list(alerts or [])
    if alerts_list:
        notes.append("; ".join(alerts_list))
        recommendations.extend(alerts_list)

    if not result.completed:
        recommendations.append("Replanejar objetivo e aumentar supervisão humana")
    if result.failures > 2:
        recommendations.append("Reduzir profundidade recursiva para estabilizar")
    if plan and plan.current_task:
        notes.append(f"Passo corrente: {plan.current_task}")

    return RetrospectiveReport(
        goal=plan.goal if plan else "",
        completed=result.completed,
        iterations=result.iterations,
        failures=result.failures,
        duration_seconds=None,
        metrics=metrics,
        recommendations=recommendations,
        notes=notes,
    )


def persist_retrospective(
    report: RetrospectiveReport,
    path: Path | str = Path("seed/memory/retrospectives.jsonl"),
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")


def load_recent_retrospectives(
    limit: int = 5,
    path: Path | str = Path("seed/memory/retrospectives.jsonl"),
) -> list[RetrospectiveReport]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    lines = path_obj.read_text(encoding="utf-8").splitlines()
    reports: list[RetrospectiveReport] = []
    for row in lines[-limit:]:
        row = row.strip()
        if not row:
            continue
        try:
            payload = json.loads(row)
        except json.JSONDecodeError:
            continue
        reports.append(RetrospectiveReport.from_dict(payload))
    return reports


__all__ = [
    "build_insight_payload",
    "Insight",
    "StatefulRetriever",
    "RetrospectiveReport",
    "build_retrospective",
    "persist_retrospective",
    "load_recent_retrospectives",
]
