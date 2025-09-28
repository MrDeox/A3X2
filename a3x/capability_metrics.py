"""Aggregation utilities to keep capability metrics in sync with run history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable


@dataclass
class _CapabilityStats:
    runs: int = 0
    completed: int = 0
    iterations_sum: float = 0.0
    apply_patch_success_sum: float = 0.0
    apply_patch_samples: int = 0
    tests_triggered: int = 0
    tests_failures: int = 0

    def register(self, completed: bool, iterations: float) -> None:
        self.runs += 1
        if completed:
            self.completed += 1
        self.iterations_sum += iterations


def compute_capability_metrics(
    log_path: Path,
) -> Dict[str, Dict[str, float | int | None]]:
    """Aggregate simple stats per capability from run evaluations."""

    stats: Dict[str, _CapabilityStats] = {}

    def ensure(capability: str) -> _CapabilityStats:
        return stats.setdefault(capability, _CapabilityStats())

    if not log_path.exists():
        return {}

    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            capabilities: Iterable[str] = record.get("capabilities") or []
            metrics: dict = record.get("metrics") or {}
            completed = bool(record.get("completed"))
            iterations = float(record.get("iterations", 0.0))

            for capability in capabilities:
                cap_stats = ensure(capability)
                cap_stats.register(completed, iterations)

                if capability == "core.diffing":
                    if "apply_patch_success_rate" in metrics:
                        cap_stats.apply_patch_success_sum += float(
                            metrics["apply_patch_success_rate"]
                        )
                        cap_stats.apply_patch_samples += 1

                if capability == "core.testing":
                    tests_count = float(metrics.get("tests_run_count", 0.0))
                    if tests_count > 0:
                        cap_stats.tests_triggered += 1
                        success_rate = float(metrics.get("tests_success_rate", 1.0))
                        if success_rate < 1.0:
                            cap_stats.tests_failures += 1

    aggregated: Dict[str, Dict[str, float | int | None]] = {}
    for capability, data in stats.items():
        cap_metrics: Dict[str, float | int | None] = {}
        if data.runs:
            completion_rate = data.completed / data.runs
            avg_iterations = data.iterations_sum / data.runs
        else:
            completion_rate = None
            avg_iterations = None

        if capability == "core.diffing":
            success_rate = None
            if data.apply_patch_samples:
                success_rate = data.apply_patch_success_sum / data.apply_patch_samples
            elif data.runs:
                success_rate = completion_rate
            cap_metrics.update(
                {
                    "success_rate": (
                        None if success_rate is None else round(success_rate, 3)
                    ),
                    "avg_iterations": (
                        None if avg_iterations is None else round(avg_iterations, 2)
                    ),
                }
            )

        elif capability == "core.testing":
            auto_trigger = None
            if data.runs:
                auto_trigger = data.tests_triggered / data.runs
            cap_metrics.update(
                {
                    "auto_trigger_rate": (
                        None if auto_trigger is None else round(auto_trigger, 3)
                    ),
                    "failures_detected": data.tests_failures,
                }
            )

        elif capability == "horiz.python":
            regression_rate = None
            if data.runs:
                regression_rate = 1.0 - (data.completed / data.runs)
            cap_metrics.update(
                {
                    "tasks_completed": data.completed,
                    "regression_rate": (
                        None if regression_rate is None else round(regression_rate, 3)
                    ),
                }
            )

        elif capability == "horiz.docs":
            cap_metrics.update(
                {
                    "docs_generated": data.completed,
                }
            )

        cap_metrics.setdefault("runs", data.runs)
        cap_metrics.setdefault("completed_runs", data.completed)
        aggregated[capability] = cap_metrics

    return aggregated


__all__ = ["compute_capability_metrics"]
