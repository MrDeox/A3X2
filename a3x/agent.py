"""Orquestrador principal do agente A3X."""

from __future__ import annotations

import json
import os
import logging
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Any

import re
import yaml
import random

from .actions import ActionType, AgentAction, AgentState, Observation
from .config import AgentConfig
from .autoeval import AutoEvaluator, EvaluationSeed
from .planner import PlannerThresholds
from .executor import ActionExecutor
from .history import AgentHistory
from .llm import BaseLLMClient, OpenRouterLLMClient
from .memory.store import SemanticMemory
from .memory.insights import build_retrospective, persist_retrospective
from .planning import HierarchicalPlanner, GoalPlan, MissionState
from .planning.storage import load_mission_state
from .policy import PolicyOverrideManager
from .llm_seed_strategist import LLMSeedStrategist


@dataclass
class AgentResult:
    completed: bool
    iterations: int
    failures: int
    history: AgentHistory
    errors: List[str]
    memories_reused: int = 0


class AgentOrchestrator:
    def __init__(
        self,
        config: AgentConfig,
        llm_client: BaseLLMClient,
        auto_evaluator: AutoEvaluator | None = None,
        depth: int = 0,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.depth = depth
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
        self.hints: Dict[str, Any] = {}
        self.action_biases: Dict[str, float] = {}
        self.logger: logging.Logger = None
        self.hints_path = self.config.workspace_root / "seed/policy_hints.json"
        self.hints = self._load_hints()
        self.recursion_depth = int(self.hints.get("recursion_depth", 3)) + int(self.hints.get("recursion_depth_adjust", 0))
        self.max_sub_depth = int(self.hints.get("max_sub_depth", 3))
        self.action_biases = self.hints.get("action_biases", {})
        self.backlog_weights = self.hints.get("backlog_weights", {})
        self._setup_logging()
        self.logger.info(f"Loaded hints: {self.hints}")
        self.logger.info(f"Applied recursion_depth: {self.recursion_depth}")
        self.logger.info(f"Applied max_sub_depth: {self.max_sub_depth}")
        self.logger.info(f"Applied action biases: {self.action_biases}")
        self.logger.info(f"Applied backlog weights: {self.backlog_weights}")
        self._semantic_memory: SemanticMemory | None = None
        self._active_plan: GoalPlan | None = None
        self._hierarchical_planner = HierarchicalPlanner(
            thresholds=thresholds,
        )
        self._policy_manager = PolicyOverrideManager()
        self._policy_manager.apply_to_agent(self)
        self._seed_strategist = LLMSeedStrategist(self.config.loop.seed_backlog)

    def _load_hints(self) -> Dict[str, Any]:
        os.makedirs(self.hints_path.parent, exist_ok=True)
        if not self.hints_path.exists():
            default_hints = {
                "action_biases": {},
                "recursion_depth": 3,
                "max_sub_depth": 3,
                "backlog_weights": {"high_delta": 1.5}
            }
            with open(self.hints_path, 'w') as f:
                json.dump(default_hints, f, indent=2)
            return default_hints
        try:
            with open(self.hints_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Fallback to defaults on error
            default_hints = {
                "action_biases": {},
                "recursion_depth": 3,
                "max_sub_depth": 3,
                "backlog_weights": {"high_delta": 1.5}
            }
            with open(self.hints_path, 'w') as f:
                json.dump(default_hints, f, indent=2)
            return default_hints

    def _setup_logging(self) -> None:
        os.makedirs(self.config.workspace_root / "a3x/logs", exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.config.workspace_root / "a3x/logs/hints.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def update_hints(self, lesson: Dict[str, Any]) -> None:
        """Append or update hints based on runtime experiences."""
        for key, value in lesson.items():
            if key in self.hints:
                if isinstance(self.hints[key], dict) and isinstance(value, dict):
                    self.hints[key].update(value)
                else:
                    self.hints[key] = value
            else:
                self.hints[key] = value
        # Atomic write
        temp_path = self.hints_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.hints, f, indent=2)
            temp_path.replace(self.hints_path)
            self.logger.info(f"Updated hints with lesson: {lesson}")
        except Exception as e:
            self.logger.error(f"Failed to update hints: {e}")
            # Fallback: try direct write
            try:
                with open(self.hints_path, 'w') as f:
                    json.dump(self.hints, f, indent=2)
            except Exception:
                pass  # Best effort

    def _propose_biased_action(self, state: AgentState) -> AgentAction:
        """Propose action with bias-weighted sampling."""
        action_types = [e.name for e in ActionType]
        prompt = f"""Based on the state, propose 3 possible next actions as a JSON list.
Each action: {{"type": "one of {action_types}", "command": "list of strings or string"}}.
State: {state}"""
        for attempt in range(3):
            try:
                response = self.llm_client.chat(prompt)
                candidates_data = json.loads(response)
                if not isinstance(candidates_data, list) or len(candidates_data) < 1:
                    raise ValueError("Invalid candidates")
                candidates = []
                for data in candidates_data[:3]:  # Limit to 3
                    atype_str = data.get('type', 'RUN_COMMAND')
                    try:
                        atype = ActionType[atype_str]
                    except KeyError:
                        continue
                    command = data.get('command', [])
                    if isinstance(command, str):
                        command = command.split()
                    action = AgentAction(type=atype, command=command)
                    candidates.append(action)
                if not candidates:
                    raise ValueError("No valid candidates")
                # Weights
                names = [c.type.name for c in candidates]
                weights = [self.action_biases.get(name, 1.0) for name in names]
                selected_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
                self.logger.info(f"Selected action {names[selected_idx]} with weights {weights}")
                return candidates[selected_idx]
            except Exception as e:
                self.logger.warning(f"Biased proposal attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    break
                time.sleep(1)
        # Fallback
        self.logger.info("Falling back to standard action proposal")
        return self.llm_client.propose_action(state)

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

        memory_lessons = self._gather_memory_lessons(goal)
        plan_alerts: List[str] = []

        # Use hierarchical planner to generate persistent objectives and subgoals
        if self._hierarchical_planner:
            # Roll forward objectives to track progress
            new_subgoals = self._hierarchical_planner.roll_forward_objectives()
            
            # If goal is a persistent objective, create appropriate missions
            for obj_id, objective in self._hierarchical_planner.objectives.items():
                if objective['description'] in goal and objective.get('status') == 'active':
                    mission = self._hierarchical_planner.create_mission_from_objective(obj_id)
                    if mission:
                        self.logger.info(f"Created mission {mission.id} for objective {obj_id}")

        # Integrate backlog weights with subgoals
        subgoals_path = self.config.workspace_root / "seed/subgoals.json"
        os.makedirs(subgoals_path.parent, exist_ok=True)
        if subgoals_path.exists():
            try:
                with open(subgoals_path, 'r') as f:
                    subgoals = json.load(f)
                # Apply weights, assuming subgoals is list of dicts with 'type' or 'delta'
                for sg in subgoals:
                    if sg.get('high_delta', False):
                        sg['priority'] = sg.get('priority', 1.0) * self.backlog_weights.get("high_delta", 1.0)
                # For queuing, e.g., sort by priority or select weighted
                # Here, log and assume Phase 1 uses it
                self.logger.info(f"Applied weights to {len(subgoals)} subgoals")
                # Optionally save back
                with open(subgoals_path, 'w') as f:
                    json.dump(subgoals, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to apply backlog weights: {e}")

        # Check if goal is complex enough to decompose using PlanComposer
        from .planning.plan_composer import PlanComposer
        plan_composer = PlanComposer()
        subtasks = plan_composer.decompose_goal(goal)
        
        # Save derived subgoals to seed/subgoals.json
        if len(subtasks) > 1:
            subgoals_data = [{"goal": st, "priority": 1.0, "parent_goal": goal} for st in subtasks]
            try:
                with open(subgoals_path, 'w') as f:
                    json.dump(subgoals_data, f, indent=2)
                self.logger.info(f"Saved {len(subtasks)} subgoals to {subgoals_path}")
            except Exception as e:
                self.logger.error(f"Failed to save subgoals: {e}")
        
        if self.depth >= self.max_sub_depth:
            self.logger.info(f"Max sub-depth {self.max_sub_depth} reached; treating as single task")
            subtasks = [goal]
        
        if len(subtasks) > 1:
            # Complex goal: use sub-agent approach
            self.logger.info(f"Decomposed goal into {len(subtasks)} subtasks: {subtasks}")
            subtask_results = plan_composer.execute_plan(subtasks, parent_agent=self)
            # Aggregate results
            output = f"Subtasks completed: {len(subtasks)}\n"
            output += f"Results: {subtask_results}\n"
            history.append(AgentAction(type=ActionType.FINISH, command=[]), Observation(success=True, output=output))
            return AgentResult(
                completed=True,
                iterations=1,  # For decomposition approach
                failures=0,
                history=history,
                errors=[],
                memories_reused=self._estimate_memories_reused(history),
            )
        else:
            # Simple goal: proceed with original loop
            self.logger.info(f"Starting main loop for goal '{goal}' with max_iterations={max_iterations}, recursion_depth={self.recursion_depth}")
            for iteration in range(1, max_iterations + 1):
                combined_context = context_summary or ""
                if memory_lessons:
                    combined_context = combined_context.strip()
                    if combined_context:
                        combined_context = f"{combined_context}\n\n{memory_lessons}"
                    else:
                        combined_context = memory_lessons

                self._active_plan = self._ensure_plan(combined_context, metrics_history)

                state = AgentState(
                    goal=goal,
                    history_snapshot=history.snapshot(),
                    iteration=iteration,
                    max_iterations=max_iterations,
                    seed_context=combined_context,
                    memory_lessons=memory_lessons,
                )

                self.logger.info(f"Iteration {iteration}/{max_iterations}: Goal='{goal[:50]}...', History events={len(history.events)}, State context length={len(str(state))}")

                action = self._propose_biased_action(state)
                self.logger.info(f"Proposed action: {action.type.name}, Command: {action.command or 'None'}")
                self._capture_llm_metrics()
                observation = self.executor.execute(action)
                self.logger.info(f"Action executed: Success={observation.success}, Output length={len(observation.output or '')}, Error={observation.error or 'None'}")
                history.append(action, observation)
                self._notify_llm(observation)

                evaluation = self._hierarchical_planner.record_action_result(
                    action,
                    observation,
                    timestamp=time.perf_counter(),
                )
                if evaluation.alerts:
                    plan_alerts.extend(evaluation.alerts)
                    for alert in evaluation.alerts:
                        print(f"[planner] {alert}")
                if evaluation.needs_replan:
                    self._active_plan = self._force_replan(state, metrics_history)

                if not observation.success:
                    failures += 1
                    if observation.error:
                        errors.append(observation.error)
                    self._seed_strategist.capture_failure(goal, action, observation)
                    if failures > self.config.limits.max_failures:
                        errors.append("Limite de falhas excedido")
                        self.logger.warning(f"Max failures {self.config.limits.max_failures} reached at iteration {iteration}")
                        break

                if (
                    action.type in {ActionType.APPLY_PATCH, ActionType.WRITE_FILE}
                    and self.config.tests.auto
                ):
                    self._run_auto_tests(history)

                if action.type is ActionType.FINISH:
                    self.logger.info(f"FINISH action proposed and executed at iteration {iteration}. Goal appears completed.")
                    result = AgentResult(
                        completed=True,
                        iterations=iteration,
                        failures=failures,
                        history=history,
                        errors=errors,
                        memories_reused=self._estimate_memories_reused(history),
                    )
                    metrics_snapshot = self._record_auto_evaluation(goal, result, started_at)
                    self._record_retrospective(result, metrics_snapshot, plan_alerts)
                    return result

            self.logger.warning(f"Loop exhausted max_iterations={max_iterations} without FINISH action. Completed=False.")
            if not errors:
                errors.append("Limite de iterações alcançado")
            result = AgentResult(
                completed=False,
                iterations=len(history.events),
                failures=failures,
                history=history,
                errors=errors,
                memories_reused=self._estimate_memories_reused(history),
            )
            metrics_snapshot = self._record_auto_evaluation(goal, result, started_at)
            self._record_retrospective(result, metrics_snapshot, plan_alerts)
            return result

    def _gather_memory_lessons(self, goal: str) -> str:
        if not self.config.loop.use_memory:
            return ""

        top_k = max(0, int(self.config.loop.memory_top_k))
        if top_k <= 0:
            return ""

        if self._semantic_memory is None:
            try:
                self._semantic_memory = SemanticMemory()
            except Exception as exc:  # pragma: no cover - falha inesperada ao carregar memória
                print(f"Memória semântica indisponível: {exc}")
                self._semantic_memory = None
                return ""

        if self._semantic_memory is None:
            return ""

        try:
            results = self._semantic_memory.query(goal, top_k=top_k)
        except Exception as exc:  # pragma: no cover - erros do backend de memória
            print(f"Erro ao consultar memória semântica: {exc}")
            return ""

        if not results:
            return ""

        snippets: List[str] = []
        for index, (entry, _score) in enumerate(results, 1):
            content = entry.content.strip()
            if len(content) > 400:
                content = content[:397] + "..."
            snippets.append(f"{index}. {entry.title}\n{content}")

        if not snippets:
            return ""

        return "Lições úteis:\n" + "\n\n".join(snippets)

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

    def _ensure_plan(
        self,
        state: AgentState,
        metrics_history: Dict[str, List[float]],
    ) -> GoalPlan | None:
        missions = self._load_missions()
        try:
            plan = self._hierarchical_planner.ensure_plan(
                state,
                missions,
                objectives=[state.goal],
                metrics_history=metrics_history,
            )
            return plan
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Planner indisponível: {exc}")
            return self._active_plan

    def _force_replan(
        self,
        state: AgentState,
        metrics_history: Dict[str, List[float]],
    ) -> GoalPlan | None:
        missions = self._load_missions()
        try:
            return self._hierarchical_planner.force_replan(
                state,
                missions,
                objectives=[state.goal],
                metrics_history=metrics_history,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Falha ao replanejar: {exc}")
            return self._active_plan

    def _load_missions(self) -> MissionState | None:
        if not self.auto_evaluator:
            return None
        missions_path = getattr(self.auto_evaluator, "missions_path", None)
        if not missions_path:
            return None
        try:
            return load_mission_state(missions_path)
        except Exception as exc:
            print(f"Não foi possível carregar missions.yaml: {exc}")
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

    def _estimate_memories_reused(self, history: AgentHistory) -> int:
        count = 0
        for event in history.events:
            for metadata in (event.action.metadata or {}, event.observation.metadata or {}):
                if not metadata:
                    continue
                if "memories_reused" in metadata:
                    try:
                        count += int(metadata.get("memories_reused", 0))
                    except (TypeError, ValueError):
                        count += 1
                elif "memory_hits" in metadata:
                    try:
                        count += int(metadata.get("memory_hits", 0))
                    except (TypeError, ValueError):
                        count += 1
                elif metadata.get("memory") == "reused":
                    count += 1
        return count

    def _record_auto_evaluation(
        self, goal: str, result: AgentResult, started_at: float
    ) -> Dict[str, float]:
        if not self.auto_evaluator:
            return {}

        duration = time.perf_counter() - started_at
        seeds: List[EvaluationSeed] = []
        metrics, inferred_caps = self._analyze_history(result)

        # Derive hints from outcomes with more sophisticated conditions
        success_rate = metrics.get("actions_success_rate", 0.0)
        if success_rate > 0.7:
            current_adjust = self.hints.get("recursion_depth_adjust", 0)
            lesson = {"recursion_depth_adjust": current_adjust + 0.5}
            self.update_hints(lesson)
        elif success_rate < 0.5:
            # Reduce depth when performance is poor
            current_adjust = self.hints.get("recursion_depth_adjust", 0)
            lesson = {"recursion_depth_adjust": max(0, current_adjust - 0.5)}
            self.update_hints(lesson)
        
        if "apply_patch_success_rate" in metrics:
            if metrics["apply_patch_success_rate"] > 0.7:
                current_bias = self.hints.get("action_biases", {}).get("APPLY_PATCH", 1.0)
                lesson = {"action_biases": {"APPLY_PATCH": current_bias + 0.1}}
                self.update_hints(lesson)
            elif metrics["apply_patch_success_rate"] < 0.5:
                # Reduce patch bias when failing
                current_bias = self.hints.get("action_biases", {}).get("APPLY_PATCH", 1.0)
                lesson = {"action_biases": {"APPLY_PATCH": max(0.1, current_bias - 0.1)}}
                self.update_hints(lesson)
        
        # Additional learning based on other metrics
        if "test_failure_rate" in metrics and metrics["test_failure_rate"] > 0.3:
            # Increase testing when failures occur
            current_bias = self.hints.get("action_biases", {}).get("RUN_COMMAND", 1.0)
            lesson = {"action_biases": {"RUN_COMMAND": current_bias + 0.2}}
            self.update_hints(lesson)
        
        # Learning from memory usage
        if "memories_reused" in metrics and metrics["memories_reused"] > 0:
            # Increase memory usage when it's effective
            current_memory_weight = self.hints.get("backlog_weights", {}).get("memory_usage", 1.0)
            lesson = {"backlog_weights": {"memory_usage": current_memory_weight + 0.1}}
            self.update_hints(lesson)
        
        # Save fitness_before and fitness_after for Phase 5
        fitness_after = self._calculate_fitness(metrics)
        fitness_before = self._get_fitness_before()  # This would need to be implemented
        delta = fitness_after - fitness_before
        
        # Store fitness data
        self._save_fitness_data(fitness_before, fitness_after, delta, goal)
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

        evaluation = self.auto_evaluator.record(
            goal=goal,
            completed=result.completed,
            iterations=result.iterations,
            failures=result.failures,
            duration_seconds=duration,
            seeds=seeds,
            metrics=metrics,
            capabilities=sorted(inferred_caps),
            errors=result.errors,
        )
        
        # Check if this was a skill creation run and register the new skill
        # This looks for specific patterns in the goal or results that indicate skill creation
        if "skill" in goal.lower() and ("create" in goal.lower() or "implement" in goal.lower()):
            self._check_and_register_new_skills()
        
        return evaluation.metrics

    def _record_retrospective(
        self,
        result: AgentResult,
        metrics_snapshot: Dict[str, float],
        plan_alerts: List[str],
    ) -> None:
        try:
            report = build_retrospective(
                result,
                self._active_plan,
                metrics_snapshot,
                alerts=plan_alerts,
            )
            persist_retrospective(report)
            self._apply_policy_overrides(report)
            self._flush_failure_seeds()
        except Exception as exc:  # pragma: no cover - defensive persistence
            print(f"Não foi possível registrar retrospectiva: {exc}")

    def _apply_policy_overrides(self, report) -> None:
        try:
            self._policy_manager.update_from_report(report, self)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Falha ao aplicar overrides de política: {exc}")

    def _flush_failure_seeds(self) -> None:
        try:
            created = self._seed_strategist.flush()
            if created:
                print(f"[seeds] {len(created)} seeds adicionadas a partir de falhas")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Não foi possível gerar seeds de falhas: {exc}")

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
        metrics["memories_reused"] = float(result.memories_reused)

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

    def _calculate_fitness(self, metrics: Dict[str, float]) -> float:
        """Calculate an overall fitness score based on metrics."""
        # Weighted combination of key metrics
        fitness = 0.0
        
        # Actions success rate (most important)
        fitness += metrics.get("actions_success_rate", 0.0) * 0.4
        
        # Apply patch success rate  
        if "apply_patch_success_rate" in metrics:
            fitness += metrics["apply_patch_success_rate"] * 0.3
        
        # Test success rate
        if "tests_success_rate" in metrics:
            fitness += metrics["tests_success_rate"] * 0.2
        
        # Recursion depth (efficiency)
        fitness += min(metrics.get("recursion_depth", 0) / 10.0, 1.0) * 0.1
        
        return fitness

    def _get_fitness_before(self) -> float:
        """Get the fitness from before this run."""
        # In a real implementation, this would load the previous fitness
        # For now, return a default value
        fitness_history_path = self.config.workspace_root / "seed/fitness_history.json"
        if fitness_history_path.exists():
            try:
                with open(fitness_history_path, 'r') as f:
                    history = json.load(f)
                    if history:
                        return history[-1].get('fitness_after', 0.0)  # Last run's fitness_after
            except Exception:
                pass
        return 0.0

    def _save_fitness_data(self, fitness_before: float, fitness_after: float, delta: float, goal: str) -> None:
        """Save fitness data to history file."""
        fitness_history_path = self.config.workspace_root / "seed/fitness_history.json"
        
        # Create directory if needed
        fitness_history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing history
        history = []
        if fitness_history_path.exists():
            try:
                with open(fitness_history_path, 'r') as f:
                    history = json.load(f)
            except Exception:
                pass  # Start with empty history if file is corrupted
        
        # Add new entry
        entry = {
            "timestamp": time.time(),
            "goal": goal,
            "fitness_before": fitness_before,
            "fitness_after": fitness_after,
            "delta": delta,
            "recursion_depth": self.recursion_depth,
            "action_biases": self.action_biases.copy(),
            "completed": True  # This would be set based on result.completed
        }
        
        history.append(entry)
        
        # Keep only last 50 entries to avoid file growing too large
        history = history[-50:]
        
        # Save back
        try:
            with open(fitness_history_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save fitness data: {e}")

    def _check_and_register_new_skills(self) -> None:
        """Check for new skills created during the run and register them."""
        from .skills_registry import get_skills_registry
        # Load the skills registry
        registry = get_skills_registry(self.config.workspace_root)
        
        # Look for new skill files in the skills directory
        for skill_file in self.config.workspace_root.glob("a3x/skills/*.py"):
            if skill_file.name.startswith("__"):
                continue
                
            # Check if this skill is already loaded
            skill_name = skill_file.stem
            if skill_name not in registry.list_skills():
                print(f"Found new skill file: {skill_file}, loading it...")
                registry.load_new_skill(str(skill_file))
                
                # Try to register the skill with an appropriate hint
                try:
                    skill_class = registry.get_skill(skill_name)
                    # Create a hint to encourage using this new skill
                    lesson = {
                        "action_biases": {skill_name: 2.0}  # Higher bias for new skills
                    }
                    self.update_hints(lesson)
                except KeyError:
                    print(f"Could not register skill: {skill_name}")

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
