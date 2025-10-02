"""Policy override helpers fed by retrospective insights."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .memory.insights import RetrospectiveReport

if TYPE_CHECKING:  # pragma: no cover
    from .agent import AgentOrchestrator


@dataclass
class PolicyOverrideManager:
    """Persist overrides derived from self-evaluation feedback."""

    path: Path = field(default_factory=lambda: Path("configs/policy_overrides.yaml"))
    data: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.data = self._load()

    # ---------------------------------------------------------------- public
    def apply_to_agent(self, agent: AgentOrchestrator) -> None:
        overrides = self.data.get("agent", {})
        recursion_depth = overrides.get("recursion_depth")
        if recursion_depth is not None:
            try:
                agent.recursion_depth = int(recursion_depth)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        max_failures = overrides.get("max_failures")
        if max_failures is not None:
            try:
                agent.config.limits.max_failures = int(max_failures)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass

    def update_from_report(self, report: RetrospectiveReport, agent: AgentOrchestrator) -> None:
        overrides = dict(self.data.get("agent", {}))
        updated = False
        for recommendation in report.recommendations:
            if "Reduzir profundidade recursiva" in recommendation:
                overrides["recursion_depth"] = max(3, agent.recursion_depth - 1)
                updated = True
            if "aumentar supervisão" in recommendation:
                overrides["max_failures"] = max(3, agent.config.limits.max_failures - 1)
                updated = True
        if updated:
            self.data["agent"] = overrides
            self._save()
            self.apply_to_agent(agent)

    # --------------------------------------------------------------- internals
    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            loaded = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return {}
        if not isinstance(loaded, dict):
            return {}
        return loaded

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.data, handle, allow_unicode=True, sort_keys=False)


__all__ = ["PolicyOverrideManager"]
