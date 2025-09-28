"""Core data structures for multi-level planning and mission tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Literal, Optional


MissionStatus = Literal["draft", "active", "paused", "completed"]
MissionPriority = Literal["moonshot", "high", "medium", "low"]
MilestoneStatus = Literal["planned", "in_progress", "blocked", "completed"]


@dataclass
class MetricSnapshot:
    """Aggregated metric view used by missions and milestones."""

    current: float
    target: Optional[float] = None
    best: Optional[float] = None
    samples: int = 0

    def to_dict(self) -> Dict[str, float | int | None]:
        return {
            "current": self.current,
            "target": self.target,
            "best": self.best,
            "samples": self.samples,
        }


@dataclass
class MissionMilestone:
    """Intermediate goals that can map directly to seeds or backlogs."""

    id: str
    goal: str
    status: MilestoneStatus = "planned"
    capability_tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    eta: Optional[str] = None  # ISO timestamp or relative horizon
    backlog_seed_id: Optional[str] = None
    metrics: Dict[str, MetricSnapshot] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status,
            "capability_tags": list(self.capability_tags),
            "dependencies": list(self.dependencies),
            "eta": self.eta,
            "backlog_seed_id": self.backlog_seed_id,
            "metrics": {name: snap.to_dict() for name, snap in self.metrics.items()},
            "notes": self.notes,
        }


@dataclass
class MissionTelemetry:
    """Long-term signals guiding planning decisions."""

    metric_summaries: Dict[str, MetricSnapshot] = field(default_factory=dict)
    reflections: List[str] = field(default_factory=list)
    discovered_gaps: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def merge_metrics(self, metrics: Iterable[tuple[str, MetricSnapshot]]) -> None:
        for name, snapshot in metrics:
            self.metric_summaries[name] = snapshot
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, object]:
        return {
            "metric_summaries": {
                name: snapshot.to_dict()
                for name, snapshot in self.metric_summaries.items()
            },
            "reflections": list(self.reflections),
            "discovered_gaps": list(self.discovered_gaps),
            "tools_required": list(self.tools_required),
            "last_updated": self.last_updated,
        }


@dataclass
class Mission:
    """Top-level mission representing a long-term autonomy objective."""

    id: str
    vision: str
    status: MissionStatus = "draft"
    priority: MissionPriority = "medium"
    success_criteria: List[str] = field(default_factory=list)
    target_metrics: Dict[str, MetricSnapshot] = field(default_factory=dict)
    capability_tags: List[str] = field(default_factory=list)
    milestones: List[MissionMilestone] = field(default_factory=list)
    telemetry: MissionTelemetry = field(default_factory=MissionTelemetry)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_milestone(self, milestone: MissionMilestone) -> None:
        if not any(existing.id == milestone.id for existing in self.milestones):
            self.milestones.append(milestone)
        self.touch()

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "vision": self.vision,
            "status": self.status,
            "priority": self.priority,
            "success_criteria": list(self.success_criteria),
            "target_metrics": {
                name: snapshot.to_dict()
                for name, snapshot in self.target_metrics.items()
            },
            "capability_tags": list(self.capability_tags),
            "milestones": [milestone.to_dict() for milestone in self.milestones],
            "telemetry": self.telemetry.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class MissionState:
    """Persistent container for missions controlled by the multi-level planner."""

    missions: List[Mission] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version: str = "0.1"

    def add_mission(self, mission: Mission) -> None:
        if any(existing.id == mission.id for existing in self.missions):
            raise ValueError(f"Mission already registered: {mission.id}")
        self.missions.append(mission)
        self.generated_at = datetime.now(timezone.utc).isoformat()

    def find_mission(self, mission_id: str) -> Optional[Mission]:
        for mission in self.missions:
            if mission.id == mission_id:
                return mission
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "missions": [mission.to_dict() for mission in self.missions],
        }


__all__ = [
    "MissionState",
    "Mission",
    "MissionMilestone",
    "MissionTelemetry",
    "MissionStatus",
    "MissionPriority",
    "MilestoneStatus",
    "MetricSnapshot",
]
