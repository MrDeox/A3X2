"""DynamicScaler for resource monitoring and adaptive scaling in A3X."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil  # For resource monitoring

from .config import AgentConfig, ScalingConfig


@dataclass
class ScalingDecision:
    """Represents a scaling decision based on resource monitoring."""
    id: str
    timestamp: str
    resource_metrics: dict[str, float]
    decision_type: str  # scale_up, scale_down, maintain
    action_taken: str  # e.g., "reduce_recursion_depth", "pause_operations"
    threshold_exceeded: dict[str, bool]
    confidence: float


class DynamicScaler:
    """Engine for dynamic resource scaling based on system monitoring."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.workspace_root = Path(config.workspace.root).resolve()
        self.scaling_path = self.workspace_root / "seed" / "scaling"
        self.scaling_path.mkdir(parents=True, exist_ok=True)
        self.decision_history: list[ScalingDecision] = self._load_scaling_history()

        # Thresholds from config
        scaling: ScalingConfig = config.scaling
        self.cpu_threshold: float = scaling.cpu_threshold
        self.memory_threshold: float = scaling.memory_threshold
        self.max_recursion_adjust: int = scaling.max_recursion_adjust
        self.current_scaling_factor: float = 1.0

    def monitor_resources(self) -> dict[str, float]:
        """Monitor current system resources using psutil."""
        metrics: dict[str, float] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent / 100.0,
            "disk_percent": psutil.disk_usage("/").percent / 100.0,
            "load_avg": psutil.getloadavg()[0] if psutil.cpu_count() else 0.0,
        }
        return metrics

    def make_scaling_decision(self, current_metrics: dict[str, float], context: Any | None = None) -> ScalingDecision:
        """Make a scaling decision based on current metrics."""
        timestamp: str = datetime.now(timezone.utc).isoformat()
        decision_id: str = f"scale_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Check thresholds
        threshold_exceeded: dict[str, bool] = {
            "cpu": current_metrics.get("cpu_percent", 0.0) > self.cpu_threshold,
            "memory": current_metrics.get("memory_percent", 0.0) > self.memory_threshold,
            "disk": current_metrics.get("disk_percent", 0.0) > 0.9,  # High disk threshold
        }

        exceeded_count: int = sum(threshold_exceeded.values())
        confidence: float = min(1.0, exceeded_count / len(threshold_exceeded))

        if exceeded_count >= 2:
            decision_type: str = "scale_down"
            action: str = "reduce_recursion_depth_or_pause"
            self.current_scaling_factor = max(0.5, self.current_scaling_factor * 0.8)
        elif exceeded_count == 1:
            decision_type = "maintain"
            action = "monitor_closely"
            self.current_scaling_factor = self.current_scaling_factor
        else:
            decision_type = "scale_up"
            action = "increase_complexity_if_possible"
            self.current_scaling_factor = min(2.0, self.current_scaling_factor * 1.2)

        decision: ScalingDecision = ScalingDecision(
            id=decision_id,
            timestamp=timestamp,
            resource_metrics=current_metrics,
            decision_type=decision_type,
            action_taken=action,
            threshold_exceeded=threshold_exceeded,
            confidence=confidence
        )

        self.decision_history.append(decision)
        self._save_scaling_decision(decision)
        self._save_scaling_history()

        # Adjust max recursion if in context
        if context and hasattr(context, "max_depth"):
            adjusted_depth: int = int(self.max_recursion_adjust * self.current_scaling_factor)
            context.max_depth = min(context.max_depth, adjusted_depth)

        return decision

    def get_scaling_recommendation(self) -> str:
        """Get a human-readable scaling recommendation."""
        recent_decisions: list[ScalingDecision] = self.decision_history[-3:]
        if not recent_decisions:
            return "No scaling history available."

        latest: ScalingDecision = recent_decisions[-1]
        if latest.decision_type == "scale_down":
            return f"Scale down operations: {latest.action_taken} (Confidence: {latest.confidence:.2f})"
        elif latest.decision_type == "scale_up":
            return f"Scale up if resources allow: {latest.action_taken} (Confidence: {latest.confidence:.2f})"
        else:
            return f"Maintain current scaling: {latest.action_taken} (Confidence: {latest.confidence:.2f})"

    def _save_scaling_decision(self, decision: ScalingDecision) -> None:
        """Save individual scaling decision to file."""
        decision_file: Path = self.scaling_path / f"{decision.id}.json"
        with decision_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(decision), f, ensure_ascii=False, indent=2)

    def _load_scaling_history(self) -> list[ScalingDecision]:
        """Load scaling history from files."""
        history: list[ScalingDecision] = []
        history_file: Path = self.scaling_path / "history.json"
        seen_ids: set[str] = set()
        duplicates_removed: bool = False

        def _add_decision(data: dict[str, Any]) -> None:
            nonlocal duplicates_removed
            try:
                decision = ScalingDecision(**data)
            except TypeError:
                return
            if decision.id in seen_ids:
                duplicates_removed = True
                return
            seen_ids.add(decision.id)
            history.append(decision)

        if history_file.exists():
            try:
                with history_file.open("r", encoding="utf-8") as f:
                    data: list[dict[str, Any]] = json.load(f)
                    for item in data:
                        if isinstance(item, dict):
                            _add_decision(item)
            except Exception:
                pass

        for decision_file in sorted(self.scaling_path.glob("scale_*.json")):
            if decision_file.name == "history.json":
                continue
            try:
                with decision_file.open("r", encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)
                    if isinstance(data, dict):
                        _add_decision(data)
            except Exception:
                pass

        if duplicates_removed and history_file.exists():
            try:
                with history_file.open("w", encoding="utf-8") as f:
                    json.dump([asdict(dec) for dec in history], f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return history

    def _save_scaling_history(self) -> None:
        """Save scaling history to file."""
        history_file: Path = self.scaling_path / "history.json"
        seen_ids: set[str] = set()
        deduplicated_history: list[ScalingDecision] = []
        for decision in self.decision_history:
            if decision.id in seen_ids:
                continue
            seen_ids.add(decision.id)
            deduplicated_history.append(decision)
        self.decision_history = deduplicated_history
        with history_file.open("w", encoding="utf-8") as f:
            json.dump([asdict(dec) for dec in self.decision_history], f, ensure_ascii=False, indent=2)

    def get_scaling_summary(self) -> dict[str, Any]:
        """Get a summary of scaling activity."""
        if not self.decision_history:
            return {"history_length": 0, "avg_confidence": 0.0, "current_factor": self.current_scaling_factor}

        avg_confidence: float = sum(dec.confidence for dec in self.decision_history) / len(self.decision_history)
        scale_up_count: int = sum(1 for dec in self.decision_history if dec.decision_type == "scale_up")
        return {
            "history_length": len(self.decision_history),
            "avg_confidence": avg_confidence,
            "scale_up_count": scale_up_count,
            "current_scaling_factor": self.current_scaling_factor,
            "recommendation": self.get_scaling_recommendation()
        }


def integrate_dynamic_scaler(config: AgentConfig) -> DynamicScaler:
    """Integrate DynamicScaler into the system."""
    scaler: DynamicScaler = DynamicScaler(config)
    # Initial monitoring
    metrics: dict[str, float] = scaler.monitor_resources()
    scaler.make_scaling_decision(metrics)
    return scaler


__all__ = [
    "DynamicScaler",
    "ScalingDecision",
    "integrate_dynamic_scaler",
]
