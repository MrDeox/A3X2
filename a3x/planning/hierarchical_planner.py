"""Hierarchical planning utilities for the autonomous loop.

This module introduces a Goal -> Mission -> Task decomposition that can be
persisted between runs.  The implementation favors explicit data classes and a
minimal persistence format (JSON) so that humans can audit or resume a plan at
any point in time.

The planner builds plans from three main sources:

* The current :class:`~a3x.actions.AgentState` (gives the goal and the latest
  context summary).
* The mission backlog (``MissionState``) maintained by the auto evaluator.
* Operational objectives configured at runtime (e.g. high-level goals chosen
  by the user or by the seed backlog).

Plans are stored under ``seed/memory/plans/<slug>.json`` together with a small
execution journal.  Every update keeps a full snapshot to simplify debugging.

The planner is intentionally deterministic so that it is easy to assert on its
behaviour during tests – we reuse the same ordering rules that the mission
planner follows and we expose a compact public API:

``ensure_plan``
    Load or build a plan for the provided state.

``record_action_result``
    Update the tracked step with the observation coming from the executor.  It
    returns a :class:`PlanEvaluation` object that callers can use to trigger a
    replan or to raise alerts.

``force_replan``
    Discard the current plan and recompute a fresh tree.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..actions import ActionType, AgentAction, AgentState, Observation
from ..constants import PERFORMANCE_DEGRADATION_THRESHOLD
from ..planner import PlannerThresholds
from ..planning.mission_state import MissionMilestone, MissionState


@dataclass
class PlanEvidence:
    """Evidence attached to a step.

    Evidence items represent objective confirmations that a step was completed
    (for instance, the output of a test command or a summary of a diff).
    """

    type: str
    description: str
    payload: str | None = None


@dataclass
class TaskPlan:
    """Leaf of the hierarchical planner."""

    id: str
    description: str
    expected_outcome: str
    metrics_target: dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending|in_progress|completed|blocked
    dependencies: list[str] = field(default_factory=list)
    evidence: list[PlanEvidence] = field(default_factory=list)
    last_observation: str | None = None
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["evidence"] = [asdict(item) for item in self.evidence]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TaskPlan:
        evidence_raw = data.get("evidence") or []
        evidence = [PlanEvidence(**item) for item in evidence_raw if isinstance(item, dict)]
        return cls(
            id=str(data.get("id")),
            description=str(data.get("description", "")),
            expected_outcome=str(data.get("expected_outcome", "")),
            metrics_target={
                str(k): float(v)
                for k, v in (data.get("metrics_target") or {}).items()
                if isinstance(v, (int, float))
            },
            status=str(data.get("status", "pending")),
            dependencies=list(data.get("dependencies", []) or []),
            evidence=evidence,
            last_observation=(
                str(data.get("last_observation")) if data.get("last_observation") else None
            ),
            blocked_reason=(
                str(data.get("blocked_reason")) if data.get("blocked_reason") else None
            ),
        )

    def mark_in_progress(self) -> None:
        if self.status == "pending":
            self.status = "in_progress"

    def mark_completed(self, observation: str, evidence: Iterable[PlanEvidence]) -> None:
        self.status = "completed"
        self.last_observation = observation
        self.evidence.extend(evidence)
        self.blocked_reason = None

    def mark_blocked(self, reason: str, observation: str | None = None) -> None:
        self.status = "blocked"
        self.blocked_reason = reason
        if observation:
            self.last_observation = observation


@dataclass
class MissionPlan:
    """A mission contains a set of ordered tasks."""

    id: str
    description: str
    priority: str
    tasks: list[TaskPlan]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MissionPlan:
        return cls(
            id=str(data.get("id")),
            description=str(data.get("description", "")),
            priority=str(data.get("priority", "medium")),
            tasks=[TaskPlan.from_dict(item) for item in data.get("tasks", []) or []],
        )

    def next_task(self) -> TaskPlan | None:
        for task in self.tasks:
            if task.status in {"pending", "in_progress"}:
                return task
        return None


@dataclass
class PlanEvent:
    timestamp: float
    step_id: str
    action_type: str
    status: str
    notes: str


@dataclass
class GoalPlan:
    """Root object describing the active plan for a goal."""

    goal: str
    missions: list[MissionPlan]
    plan_id: str
    metrics_snapshot: dict[str, float] = field(default_factory=dict)
    current_mission: str | None = None
    current_task: str | None = None
    events: list[PlanEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "missions": [mission.to_dict() for mission in self.missions],
            "plan_id": self.plan_id,
            "metrics_snapshot": self.metrics_snapshot,
            "current_mission": self.current_mission,
            "current_task": self.current_task,
            "events": [asdict(event) for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GoalPlan:
        events = [PlanEvent(**item) for item in data.get("events", []) or [] if isinstance(item, dict)]
        return cls(
            goal=str(data.get("goal", "")),
            missions=[MissionPlan.from_dict(item) for item in data.get("missions", []) or []],
            plan_id=str(data.get("plan_id", "")),
            metrics_snapshot={
                str(k): float(v)
                for k, v in (data.get("metrics_snapshot") or {}).items()
                if isinstance(v, (int, float))
            },
            current_mission=(
                str(data.get("current_mission")) if data.get("current_mission") else None
            ),
            current_task=(
                str(data.get("current_task")) if data.get("current_task") else None
            ),
            events=events,
        )

    def locate_step(self, step_id: str | None) -> TaskPlan | None:
        if step_id is None:
            return None
        for mission in self.missions:
            for task in mission.tasks:
                if task.id == step_id:
                    return task
        return None

    def update_current_step(self) -> None:
        for mission in self.missions:
            task = mission.next_task()
            if task:
                self.current_mission = mission.id
                self.current_task = task.id
                task.mark_in_progress()
                return
        self.current_mission = None
        self.current_task = None


@dataclass
class PlanEvaluation:
    """Result of evaluating an action against the plan."""

    needs_replan: bool = False
    alerts: list[str] = field(default_factory=list)

    def register_alert(self, message: str) -> None:
        self.alerts.append(message)
        if "replanejamento" in message or "falhou" in message:
            self.needs_replan = True


def _slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-") or "goal"


class HierarchicalPlanner:
    """Main entry-point for hierarchical planning."""

    def __init__(
        self,
        storage_dir: Path | str = Path("seed/memory/plans"),
        *,
        thresholds: PlannerThresholds | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds or PlannerThresholds()
        self.objectives = {}
        self._cached_plan: GoalPlan | None = None

    # ------------------------------------------------------------------ public
    def ensure_plan(
        self,
        state: AgentState,
        missions: MissionState | None,
        objectives: Sequence[str],
        metrics_history: dict[str, list[float]],
    ) -> GoalPlan:
        plan_id = _slugify(state.goal)
        plan = self._load_plan(plan_id)
        if plan is None or self._metrics_degraded(plan, metrics_history):
            plan = self._build_plan(state, missions, objectives, metrics_history)
            self._save_plan(plan)
        plan.update_current_step()
        self._cached_plan = plan
        return plan

    def force_replan(
        self,
        state: AgentState,
        missions: MissionState | None,
        objectives: Sequence[str],
        metrics_history: dict[str, list[float]],
    ) -> GoalPlan:
        plan = self._build_plan(state, missions, objectives, metrics_history)
        self._save_plan(plan)
        self._cached_plan = plan
        return plan

    def record_action_result(
        self,
        action: AgentAction,
        observation: Observation,
        timestamp: float,
    ) -> PlanEvaluation:
        if self._cached_plan is None:
            return PlanEvaluation()
        plan = self._cached_plan
        step = plan.locate_step(plan.current_task)
        evaluation = PlanEvaluation()
        if step is None:
            evaluation.register_alert("Nenhum passo ativo para avaliar – replanejamento sugerido.")
            return evaluation

        observation_excerpt = (observation.output or "").strip()
        if len(observation_excerpt) > 400:
            observation_excerpt = observation_excerpt[:397] + "..."

        if observation.success:
            evidence = self._build_evidence(action, observation_excerpt)
            step.mark_completed(observation_excerpt, evidence)
        else:
            reason = observation.error or "Ação falhou sem descrição."
            step.mark_blocked(reason, observation_excerpt)
            evaluation.register_alert(f"Passo {step.id} falhou: {reason}. replanejamento sugerido")

        plan.events.append(
            PlanEvent(
                timestamp=timestamp,
                step_id=step.id,
                action_type=action.type.name,
                status=step.status,
                notes=observation.error or observation_excerpt or "sem saída",
            )
        )
        plan.update_current_step()
        self._save_plan(plan)
        return evaluation

    def current_plan(self) -> GoalPlan | None:
        return self._cached_plan

    # ----------------------------------------------------------------- helpers
    def _build_plan(
        self,
        state: AgentState,
        missions: MissionState | None,
        objectives: Sequence[str],
        metrics_history: dict[str, list[float]],
    ) -> GoalPlan:
        mission_plans: list[MissionPlan] = []
        if missions:
            for mission in missions.missions:
                description = mission.vision or f"Missão {mission.id}"
                mission_plan = MissionPlan(
                    id=str(mission.id),
                    description=description,
                    priority=mission.priority,
                    tasks=self._convert_milestones(mission.milestones),
                )
                mission_plans.append(mission_plan)

        if not mission_plans:
            # Fallback: create a minimal mission with generic steps so that the
            # orchestrator always has guidance.
            fallback_tasks = self._build_fallback_tasks(state.goal)
            mission_plans.append(
                MissionPlan(
                    id="fallback",
                    description="Explorar objetivo corrente",
                    priority="medium",
                    tasks=fallback_tasks,
                )
            )

        plan_id = _slugify(state.goal)
        plan = GoalPlan(
            goal=state.goal,
            missions=mission_plans,
            plan_id=plan_id,
            metrics_snapshot=self._latest_metrics(metrics_history),
        )
        plan.update_current_step()
        return plan

    def _convert_milestones(self, milestones: list[MissionMilestone]) -> list[TaskPlan]:
        tasks: list[TaskPlan] = []
        for milestone in milestones:
            if milestone.status == "completed":
                continue
            task_id = f"{milestone.id}"
            description = milestone.goal or f"Milestone {milestone.id}"
            metrics = {
                metric: snapshot.target
                for metric, snapshot in milestone.metrics.items()
                if snapshot.target is not None
            }
            expected = textwrap.dedent(

                    milestone.notes
                    or "Preparar alteração, implementar e validar com testes automatizados."

            ).strip()
            tasks.append(
                TaskPlan(
                    id=task_id,
                    description=description,
                    expected_outcome=expected,
                    metrics_target=metrics,
                    dependencies=list(milestone.dependencies or []),
                )
            )
        return tasks

    def _build_fallback_tasks(self, goal: str) -> list[TaskPlan]:
        return [
            TaskPlan(
                id="analisar",
                description="Analisar contexto e estabelecer hipóteses",
                expected_outcome=f"Entendimento claro do objetivo '{goal}' e plano de execução",
            ),
            TaskPlan(
                id="executar",
                description="Implementar mudanças principais",
                expected_outcome="Alterações aplicadas com diffs revisados",
                dependencies=["analisar"],
            ),
            TaskPlan(
                id="validar",
                description="Executar testes e validações",
                expected_outcome="Resultados de testes documentados",
                dependencies=["executar"],
            ),
            TaskPlan(
                id="documentar",
                description="Documentar aprendizados",
                expected_outcome="Insights e documentação atualizados",
                dependencies=["validar"],
            ),
        ]

    def _latest_metrics(self, metrics_history: dict[str, list[float]]) -> dict[str, float]:
        snapshot: dict[str, float] = {}
        for name, values in metrics_history.items():
            if values:
                snapshot[name] = float(values[-1])
        return snapshot

    def _metrics_degraded(
        self, plan: GoalPlan, metrics_history: dict[str, list[float]]
    ) -> bool:
        if not plan.metrics_snapshot:
            return False
        for name, baseline in plan.metrics_snapshot.items():
            latest_values = metrics_history.get(name)
            if not latest_values:
                continue
            latest = latest_values[-1]
            if baseline == 0:
                continue
            variation = (baseline - latest) / abs(baseline)
            if variation >= PERFORMANCE_DEGRADATION_THRESHOLD:  # drop of configured % from baseline triggers replan
                return True
        return False

    def _build_evidence(
        self, action: AgentAction, observation_excerpt: str
    ) -> list[PlanEvidence]:
        evidence: list[PlanEvidence] = []
        if action.type == ActionType.RUN_COMMAND:
            evidence.append(
                PlanEvidence(
                    type="command_output",
                    description=f"Comando '{action.command}' executado",
                    payload=observation_excerpt,
                )
            )
        elif action.type == ActionType.APPLY_PATCH:
            evidence.append(
                PlanEvidence(
                    type="patch_applied",
                    description="Patch aplicado com sucesso",
                    payload=observation_excerpt,
                )
            )
        elif action.type == ActionType.WRITE_FILE:
            evidence.append(
                PlanEvidence(
                    type="file_written",
                    description=f"Arquivo {action.path} atualizado",
                    payload=observation_excerpt,
                )
            )
        elif action.type == ActionType.FINISH:
            evidence.append(
                PlanEvidence(
                    type="finish",
                    description="Fluxo encerrado pelo LLM",
                    payload=observation_excerpt,
                )
            )
        return evidence

    # --------------------------------------------------------------- persistence
    def _plan_path(self, plan_id: str) -> Path:
        return self.storage_dir / f"{plan_id}.json"

    def _load_plan(self, plan_id: str) -> GoalPlan | None:
        path = self._plan_path(plan_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        plan = GoalPlan.from_dict(raw)
        return plan

    def _save_plan(self, plan: GoalPlan) -> None:
        path = self._plan_path(plan.plan_id)
        payload = plan.to_dict()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def roll_forward_objectives(self):
        """Roll forward objectives from storage or return current."""
        if self._cached_plan:
            return [mission.description for mission in self._cached_plan.missions]
        return []


__all__ = [
    "PlanEvidence",
    "TaskPlan",
    "MissionPlan",
    "PlanEvent",
    "GoalPlan",
    "PlanEvaluation",
    "HierarchicalPlanner",
]
