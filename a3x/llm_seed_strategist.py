"""Generate backlog seeds directly from runtime failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path

from .actions import AgentAction, Observation
from .seeds import Seed, SeedBacklog


@dataclass
class FailureEvent:
    goal: str
    action_type: str
    description: str
    snippet: str

    def seed_id(self) -> str:
        digest = md5(f"{self.goal}|{self.action_type}|{self.description}".encode()).hexdigest()
        return f"auto-failure.{digest[:10]}"


@dataclass
class LLMSeedStrategist:
    """Track executor failures and turn them into actionable seeds."""

    backlog_path: Path
    _events: list[FailureEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.backlog_path = Path(self.backlog_path)

    def capture_failure(self, goal: str, action: AgentAction, observation: Observation) -> None:
        snippet = (observation.output or observation.error or "").strip()
        if len(snippet) > 400:
            snippet = snippet[:397] + "..."
        description = observation.error or "Falha sem descrição"
        event = FailureEvent(
            goal=goal,
            action_type=action.type.name,
            description=description,
            snippet=snippet,
        )
        self._events.append(event)

    def flush(self) -> list[Seed]:
        if not self._events:
            return []
        backlog = SeedBacklog.load(self.backlog_path)
        created: list[Seed] = []
        for event in self._events:
            seed_id = event.seed_id()
            if backlog.exists(seed_id):
                continue
            seed = Seed(
                id=seed_id,
                goal=f"Investigar falha: {event.description}",
                priority="high",
                status="pending",
                type="failure",
                metadata={
                    "source": "llm_failure_capture",
                    "action_type": event.action_type,
                    "goal": event.goal,
                    "snippet": event.snippet,
                },
            )
            backlog.add_seed(seed)
            created.append(seed)
        backlog.save()
        self._events.clear()
        return created


__all__ = ["LLMSeedStrategist"]
