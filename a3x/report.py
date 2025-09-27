"""SeedAI reporting utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from .capabilities import CapabilityRegistry


@dataclass
class CapabilityUsage:
    capability_id: str
    name: str
    category: str
    runs: int = 0
    completed_runs: int = 0

    def register(self, completed: bool) -> None:
        self.runs += 1
        if completed:
            self.completed_runs += 1

    @property
    def completion_rate(self) -> float:
        if not self.runs:
            return 0.0
        return self.completed_runs / self.runs


def generate_capability_report(
    capabilities_path: Path | str = Path("seed/capabilities.yaml"),
    evaluations_log: Path | str = Path("seed/evaluations/run_evaluations.jsonl"),
    metrics_history: Path | str = Path("seed/metrics/history.json"),
    output_path: Path | str = Path("seed/reports/capability_report.md"),
) -> None:
    capabilities_path = Path(capabilities_path)
    evaluations_log = Path(evaluations_log)
    metrics_history = Path(metrics_history)
    output_path = Path(output_path)

    if not capabilities_path.exists():
        return

    registry = CapabilityRegistry.from_yaml(capabilities_path)
    usage_map: Dict[str, CapabilityUsage] = {
        cap.id: CapabilityUsage(capability_id=cap.id, name=cap.name, category=cap.category)
        for cap in registry.list()
    }

    metrics_summary = _load_metrics_summary(metrics_history)
    evaluation_count = 0
    completed_count = 0

    if evaluations_log.exists():
        with evaluations_log.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                evaluation_count += 1
                if record.get("completed"):
                    completed_count += 1
                for capability_id in record.get("capabilities", []):
                    usage = usage_map.get(capability_id)
                    if usage:
                        usage.register(bool(record.get("completed")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = _render_report(usage_map.values(), evaluation_count, completed_count, metrics_summary)
    output_path.write_text(report, encoding="utf-8")


def _load_metrics_summary(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    history = json.loads(path.read_text(encoding="utf-8"))
    summary: Dict[str, Dict[str, float]] = {}
    for metric, values in history.items():
        if not values:
            continue
        summary[metric] = {
            "last": float(values[-1]),
            "best": float(max(values)),
            "count": float(len(values)),
        }
    return summary


def _render_report(
    usage: Iterable[CapabilityUsage],
    evaluation_count: int,
    completed_count: int,
    metrics_summary: Dict[str, Dict[str, float]],
) -> str:
    lines = ["# SeedAI Capability Report"]
    lines.append("")
    lines.append(f"Total evaluations: {evaluation_count} | Completed: {completed_count}")
    if evaluation_count:
        completion_rate = completed_count / evaluation_count
        lines.append(f"Overall completion rate: {completion_rate:.2%}")
    lines.append("")

    lines.append("## Capability Usage")
    lines.append("")
    lines.append("| Capability | Category | Runs | Completion Rate |")
    lines.append("|------------|----------|------|------------------|")
    for item in usage:
        if item.runs == 0:
            continue
        lines.append(
            f"| {item.name} ({item.capability_id}) | {item.category} | {item.runs} | {item.completion_rate:.2%} |"
        )
    if len(lines) == 6:
        lines.append("Nenhum uso registrado ainda.")
    lines.append("")

    if metrics_summary:
        lines.append("## Metrics Summary")
        lines.append("")
        lines.append("| Metric | Best | Last | Samples |")
        lines.append("|--------|------|------|---------|")
        for name, data in metrics_summary.items():
            lines.append(
                f"| {name} | {data['best']:.4f} | {data['last']:.4f} | {int(data['count'])} |"
            )
    else:
        lines.append("## Metrics Summary")
        lines.append("")
        lines.append("Nenhuma métrica registrada ainda.")

    lines.append("")
    lines.append("Relatório gerado automaticamente pelo A3X SeedAI.")
    return "\n".join(lines)


__all__ = ["generate_capability_report"]

