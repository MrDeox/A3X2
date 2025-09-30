"""Orquestrador principal do agente A3X."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import re

from .actions import ActionType, AgentAction, AgentState, Observation
from .config import AgentConfig
from .autoeval import AutoEvaluator, EvaluationSeed
from .planner import PlannerThresholds
from .executor import ActionExecutor
from .history import AgentHistory
from .llm import BaseLLMClient


@dataclass
class AgentResult:
    completed: bool
    iterations: int
    failures: int
    history: AgentHistory
    errors: List[str]


class AgentOrchestrator:
    def __init__(
        self,
        config: AgentConfig,
        llm_client: BaseLLMClient,
        auto_evaluator: AutoEvaluator | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.executor = ActionExecutor(config)
        thresholds = PlannerThresholds(
            apply_patch_success_rate=config.goals.get_threshold(
                "apply_patch_success_rate", 0.8
            ),
            actions_success_rate=config.goals.get_threshold(
                "actions_success_rate", 0.8
            ),
            tests_success_rate=config.goals.get_threshold("tests_success_rate", 0.9),
        )
        self.auto_evaluator = auto_evaluator or AutoEvaluator(thresholds=thresholds, config=config)
        self._llm_metrics: Dict[str, List[float]] = {}
        self.recursion_depth: int = 3

    def run(self, goal: str) -> AgentResult:
        history = AgentHistory()
        self.llm_client.start(goal)

        failures = 0
        errors: List[str] = []
        # Dynamically adjust recursion_depth based on previous success_rate
        metrics_history = self.auto_evaluator._read_metrics_history()
        actions_rates = metrics_history.get("actions_success_rate", [0.0])
        avg_success_rate = sum(actions_rates[-3:]) / len(actions_rates[-3:]) if len(actions_rates) >= 3 else actions_rates[-1] if actions_rates else 0.0
        if avg_success_rate > 0.85:
            self.recursion_depth = min(10, self.recursion_depth + 1)
        elif avg_success_rate < 0.6:
            self.recursion_depth = max(3, self.recursion_depth - 1)
        # Stabilize at higher depth for target >=5
        if self.recursion_depth < 5 and avg_success_rate > 0.8:
            self.recursion_depth = 5

        print("Debug: Agent running iteration")

        base_iterations = 10  # Base iterations per recursion level
        max_iterations = base_iterations * self.recursion_depth
        started_at = time.perf_counter()
        context_summary = self.auto_evaluator.latest_summary()

        for iteration in range(1, max_iterations + 1):
            state = AgentState(
                goal=goal,
                history_snapshot=history.snapshot(),
                iteration=iteration,
                max_iterations=max_iterations,
                seed_context=context_summary,
            )

            action = self.llm_client.propose_action(state)
            self._capture_llm_metrics()
            observation = self.executor.execute(action)
            history.append(action, observation)
            self._notify_llm(observation)

            if not observation.success:
                failures += 1
                if failures > self.config.limits.max_failures:
                    errors.append("Limite de falhas excedido")
                    break

            if (
                action.type in {ActionType.APPLY_PATCH, ActionType.WRITE_FILE}
                and self.config.tests.auto
            ):
                self._run_auto_tests(history)

            if action.type is ActionType.FINISH:
                result = AgentResult(
                    completed=True,
                    iterations=iteration,
                    failures=failures,
                    history=history,
                    errors=errors,
                )
                self._record_auto_evaluation(goal, result, started_at)
                return result

        if not errors:
            errors.append("Limite de iterações alcançado")
        result = AgentResult(
            completed=False,
            iterations=len(history.events),
            failures=failures,
            history=history,
            errors=errors,
        )
        self._record_auto_evaluation(goal, result, started_at)
        return result

    # Internos -----------------------------------------------------------------

    def _run_auto_tests(self, history: AgentHistory) -> None:
        for command in self.config.tests.commands:
            action = AgentAction(type=ActionType.RUN_COMMAND, command=command)
            observation = self.executor.execute(action)
            history.append(action, observation)
            self._notify_llm(observation)

    def _notify_llm(self, observation: Observation) -> None:
        excerpt = observation.output
        if len(excerpt) > 2_000:
            excerpt = excerpt[:1_997] + "..."
        self.llm_client.notify_observation(excerpt)

    def _capture_llm_metrics(self) -> None:
        metrics = self.llm_client.get_last_metrics()
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._llm_metrics.setdefault(key, []).append(float(value))

    def _aggregate_llm_metrics(self) -> Dict[str, float]:
        aggregated: Dict[str, float] = {}
        for key, values in self._llm_metrics.items():
            if not values:
                continue
            aggregated[f"{key}_last"] = values[-1]
            aggregated[f"{key}_avg"] = sum(values) / len(values)
        return aggregated

    def _record_auto_evaluation(
        self, goal: str, result: AgentResult, started_at: float
    ) -> None:
        if not self.auto_evaluator:
            return

        duration = time.perf_counter() - started_at
        seeds: List[EvaluationSeed] = []
        metrics, inferred_caps = self._analyze_history(result)
        metrics.update(self._aggregate_llm_metrics())
        if not result.completed:
            seeds.append(
                EvaluationSeed(
                    description="Investigar por que o objetivo não foi concluído.",
                    priority="high",
                    seed_type="analysis",
                )
            )
        if result.failures:
            seeds.append(
                EvaluationSeed(
                    description="Reduzir falhas durante execução (avaliar logs e políticas).",
                    priority="medium",
                    seed_type="analysis",
                )
            )
        if metrics.get("actions_success_rate", 1.0) < 0.8:
            seeds.append(
                EvaluationSeed(
                    description="Melhorar taxa de sucesso das ações (ajustar prompts ou políticas).",
                    priority="medium",
                    seed_type="analysis",
                )
            )
        if (
            metrics.get("apply_patch_success_rate") is not None
            and metrics["apply_patch_success_rate"] < 1.0
        ):
            seeds.append(
                EvaluationSeed(
                    description="Investigar falhas ao aplicar patches e aprimorar heurísticas de edição.",
                    priority="medium",
                    capability="core.diffing",
                    seed_type="analysis",
                )
            )

        metric_test_targets = {
            "actions_success_rate": None,
            "apply_patch_success_rate": "core.diffing",
        }
        for metric_name, capability in metric_test_targets.items():
            if metric_name in metrics:
                seeds.append(
                    EvaluationSeed(
                        description=f"Garantir teste automatizado para métrica {metric_name}.",
                        priority="low",
                        capability=capability,
                        seed_type="test",
                        data={"metric": metric_name},
                    )
                )

        self.auto_evaluator.record(
            goal=goal,
            completed=result.completed,
            iterations=result.iterations,
            failures=result.failures,
            duration_seconds=duration,
            seeds=seeds,
            metrics=metrics,
            capabilities=sorted(inferred_caps),
        )

    def _analyze_history(
        self, result: AgentResult
    ) -> tuple[Dict[str, float], Set[str]]:
        events = result.history.events
        total_actions = len(events)
        success_actions = sum(1 for event in events if event.observation.success)
        metrics: Dict[str, float] = {}
        if total_actions:
            metrics["actions_total"] = float(total_actions)
            metrics["actions_success_rate"] = success_actions / total_actions
        else:
            metrics["actions_total"] = 0.0
            metrics["actions_success_rate"] = 0.0

        apply_patch_events = [
            event for event in events if event.action.type is ActionType.APPLY_PATCH
        ]
        if apply_patch_events:
            success_count = sum(
                1 for event in apply_patch_events if event.observation.success
            )
            metrics["apply_patch_count"] = float(len(apply_patch_events))
            metrics["apply_patch_success_rate"] = success_count / len(
                apply_patch_events
            )
        else:
            metrics["apply_patch_count"] = 0.0

        unique_commands: Set[str] = set()
        for event in events:
            if event.action.type is ActionType.RUN_COMMAND and event.action.command:
                unique_commands.add(event.action.command[0])
        metrics["unique_commands"] = float(len(unique_commands))

        file_extensions: Set[str] = set()
        for event in events:
            ext = _infer_extension(event.action)
            if ext:
                file_extensions.add(ext)
        metrics["unique_file_extensions"] = float(len(file_extensions))

        inferred_capabilities: Set[str] = set()
        if apply_patch_events:
            inferred_capabilities.add("core.diffing")
        if any(cmd in {"pytest", "ruff", "black"} for cmd in unique_commands):
            inferred_capabilities.add("core.testing")
            inferred_capabilities.add("horiz.python")
        if any(ext == "py" for ext in file_extensions):
            inferred_capabilities.add("horiz.python")
        if any(ext in {"md", "rst"} for ext in file_extensions):
            inferred_capabilities.add("horiz.docs")

        metrics["failures"] = float(result.failures)
        metrics["iterations"] = float(result.iterations)
        metrics["recursion_depth"] = float(self.recursion_depth)

        test_runs = [
            event
            for event in events
            if event.action.type is ActionType.RUN_COMMAND
            and event.action.command
            and any("pytest" in part for part in event.action.command)
        ]
        if test_runs:
            success = sum(1 for event in test_runs if event.observation.success)
            metrics["tests_run_count"] = float(len(test_runs))
            metrics["tests_success_rate"] = success / len(test_runs)
        else:
            metrics["tests_run_count"] = 0.0

        lint_commands = [
            event
            for event in events
            if event.action.type is ActionType.RUN_COMMAND
            and event.action.command
            and any(cmd in " ".join(event.action.command) for cmd in ("ruff", "black"))
        ]
        if lint_commands:
            success = sum(1 for event in lint_commands if event.observation.success)
            metrics["lint_run_count"] = float(len(lint_commands))
            metrics["lint_success_rate"] = success / len(lint_commands)
        else:
            metrics["lint_run_count"] = 0.0

        return metrics, inferred_capabilities


_PATCH_FILE_PATTERN = re.compile(r"^\+\+\+\s+[ab]/(?P<path>.+)$", re.MULTILINE)


def _infer_extension(action: AgentAction) -> str | None:
    if action.type is ActionType.WRITE_FILE and action.path:
        return Path(action.path).suffix.lstrip(".") or None
    if action.type is ActionType.APPLY_PATCH and action.diff:
        match = _PATCH_FILE_PATTERN.search(action.diff)
        if match:
            path = match.group("path")
            return Path(path).suffix.lstrip(".") or None
    return None

    def _capture_llm_metrics(self) -> None:
        metrics = self.llm_client.get_last_metrics()
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._llm_metrics.setdefault(key, []).append(float(value))

    def _aggregate_llm_metrics(self) -> Dict[str, float]:
        aggregated: Dict[str, float] = {}
        for key, values in self._llm_metrics.items():
            if not values:
                continue
            aggregated[f"{key}_last"] = values[-1]
            aggregated[f"{key}_avg"] = sum(values) / len(values)
        return aggregated
