"""Utilities to convert evaluations into semantic memory entries."""

from __future__ import annotations

from typing import Dict, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..autoeval import RunEvaluation


def build_insight_payload(
    evaluation: "RunEvaluation",
    capability_metrics: Optional[Dict[str, Dict[str, float | int | None]]] = None,
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
    return title, content, tags, metadata
