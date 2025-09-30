"""Utilities to convert evaluations into semantic memory entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from hashlib import md5

from ..actions import AgentState
from .store import MemoryEntry, SemanticMemory

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..autoeval import RunEvaluation


def build_insight_payload(
    evaluation: "RunEvaluation",
    capability_metrics: Optional[Dict[str, Dict[str, float | int | None]]] = None,
    snapshot: Optional[str] = None,
) -> tuple[str, str, List[str], dict]:
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
        cap_lines: List[str] = []
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
    tags: List[str]
    metadata: dict
    created_at: str
    similarity: float = 0.0

    @classmethod
    def from_entry(cls, entry: MemoryEntry, similarity: float) -> "Insight":
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
        self.last_snapshot_hash: Optional[str] = None
        self.derivation_threshold = 0.1  # Simple diff ratio threshold for flag
        self.similarity_threshold = 0.5  # Lowered for better retrieval in structured content

    def retrieve_session_context(self, state: AgentState) -> List[Insight]:
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
