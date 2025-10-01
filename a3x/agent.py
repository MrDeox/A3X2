"""Main orchestrator for the A3X agent."""

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
    # Constants for magic numbers
    DEFAULT_RECURSION_DEPTH = 3
    DEFAULT_MAX_SUB_DEPTH = 3
    HIGH_SUCCESS_THRESHOLD = 0.85
    LOW_SUCCESS_THRESHOLD = 0.6
    MIN_RECURSION_DEPTH = 3
    MAX_RECURSION_DEPTH = 10
    STABLE_RECURSION_DEPTH = 5
    HIGH_STABLE_SUCCESS_THRESHOLD = 0.8
    BASE_ITERATIONS = 10
    MEMORY_SNIPPET_MAX_LENGTH = 400
    MEMORY_ELLIPSIS_LENGTH = 397
    OBSERVATION_EXCERPT_MAX_LENGTH = 2000
    OBSERVATION_ELLIPSIS_LENGTH = 1997
    HIGH_SUCCESS_RATE = 0.7
    LOW_SUCCESS_RATE = 0.5
    PATCH_HIGH_SUCCESS_RATE = 0.7
    PATCH_LOW_SUCCESS_RATE = 0.5
    TEST_FAILURE_THRESHOLD = 0.3
    FITNESS_ACTIONS_WEIGHT = 0.4
    FITNESS_PATCH_WEIGHT = 0.3
    FITNESS_TESTS_WEIGHT = 0.2
    FITNESS_RECURSION_WEIGHT = 0.1
    MAX_RECURSION_FOR_FITNESS = 10
    BACKLOG_HIGH_DELTA_WEIGHT = 1.5
    DEFAULT_BACKLOG_MEMORY_USAGE_WEIGHT = 1.0
    NEW_SKILL_BIAS = 2.0
    MAX_FITNESS_HISTORY_ENTRIES = 50
    LLM_PROPOSAL_ATTEMPTS = 3
    BIAS_PROPOSAL_CANDIDATES = 3
    MEMORY_TOP_K_DEFAULT = 0  # Config-driven, but default for checks
    DEFAULT_HINTS_ACTION_BIASES = {}
    DEFAULT_HINTS_RECURSION_DEPTH = 3
    DEFAULT_HINTS_MAX_SUB_DEPTH = 3
    DEFAULT_HINTS_BACKLOG_WEIGHTS = {"high_delta": BACKLOG_HIGH_DELTA_WEIGHT}

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
        self.recursion_depth = int(self.hints.get("recursion_depth", self.DEFAULT_RECURSION_DEPTH)) + int(self.hints.get("recursion_depth_adjust", 0))
        self.max_sub_depth = int(self.hints.get("max_sub_depth", self.DEFAULT_MAX_SUB_DEPTH))
        self.action_biases = self.hints.get("action_biases", self.DEFAULT_HINTS_ACTION_BIASES)
        self.backlog_weights = self.hints.get("backlog_weights", self.DEFAULT_HINTS_BACKLOG_WEIGHTS)
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
        """Load hints from policy_hints.json or create defaults."""
        os.makedirs(self.hints_path.parent, exist_ok=True)
        if not self.hints_path.exists():
            default_hints = {
                "action_biases": self.DEFAULT_HINTS_ACTION_BIASES,
                "recursion_depth": self.DEFAULT_RECURSION_DEPTH,
                "max_sub_depth": self.DEFAULT_MAX_SUB_DEPTH,
                "backlog_weights": self.DEFAULT_HINTS_BACKLOG_WEIGHTS
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
                "action_biases": self.DEFAULT_HINTS_ACTION_BIASES,
                "recursion_depth": self.DEFAULT_RECURSION_DEPTH,
                "max_sub_depth": self.DEFAULT_MAX_SUB_DEPTH,
                "backlog_weights": self.DEFAULT_HINTS_BACKLOG_WEIGHTS
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
        """Propose an action with bias-weighted sampling from LLM candidates.

        Args:
            state: Current agent state including goal, history, and context.

        Returns:
            AgentAction: The selected biased action.
        """
        action_types = [e.name for e in ActionType]
        prompt = f"""Based on the state, propose {self.BIAS_PROPOSAL_CANDIDATES} possible next actions as a JSON list.
Each action: {{"type": "one of {action_types}", "command": "list of strings or string"}}.
State: {state}"""
        for attempt in range(self.LLM_PROPOSAL_ATTEMPTS):
            try:
                response = self.llm_client.chat(prompt)
                candidates_data = json.loads(response)
                if not isinstance(candidates_data, list) or len(candidates_data) < 1:
                    raise ValueError("Invalid candidates")
                candidates = []
                for data in candidates_data[:self.BIAS_PROPOSAL_CANDIDATES]:  # Limit to constant
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
                if attempt == self.LLM_PROPOSAL_ATTEMPTS - 1:
                    break
                time.sleep(1)
        # Fallback
        self.logger.info("Falling back to standard action proposal")
        return self.llm_client.propose_action(state)

    def run(self, goal: str) -> AgentResult:
        """Execute the agent's main loop to achieve the given goal.

        This method orchestrates the hierarchical planning, action execution, and evaluation
        to fulfill the specified goal. It handles goal decomposition into subtasks if needed,
        runs the iteration loop, and records results including metrics and retrospectives.

        Args:
            goal (str): The objective string describing what the agent should accomplish.

        Returns:
            AgentResult: A summary object containing completion status, iteration count,
                failure count, execution history, errors encountered, and memories reused.

        Examples:
            result = agent.run("Implement a new feature in the codebase")
            if result.completed:
                print("Goal achieved successfully.")
            else:
                print(f"Goal incomplete after {result.iterations} iterations.")
        """
        history = AgentHistory()
        self.llm_client.start(goal)

        failures = 0
        errors: List[str] = []

        # Adjust recursion depth based on historical success rate
        self._adjust_recursion_depth()

        print("Debug: Agent running iteration")

        max_iterations = self.BASE_ITERATIONS * self.recursion_depth
        started_at = time.perf_counter()
        context_summary = self.auto_evaluator.latest_summary()

        memory_lessons = self._gather_memory_lessons(goal)
        plan_alerts: List[str] = []

        # Hierarchical planning setup
        self._setup_hierarchical_planning(goal)

        # Integrate backlog weights with subgoals
        self._apply_backlog_weights_to_subgoals()

        # Decompose goal if complex
        subtasks = self._decompose_goal(goal)

        if self.depth >= self.max_sub_depth:
            self.logger.info(f"Max sub-depth {self.max_sub_depth} reached; treating as single task")
            subtasks = [goal]

        if len(subtasks) > 1:
            # Handle complex goal with sub-agent approach
            return self._handle_subtasks(subtasks, history, goal)
        else:
            # Run main loop for simple goal
            metrics_history = self.auto_evaluator._read_metrics_history()
            return self._run_main_loop(goal, history, failures, errors, max_iterations, context_summary, memory_lessons, metrics_history, started_at, plan_alerts)

    def _adjust_recursion_depth(self) -> None:
        """Dynamically adjust recursion_depth based on previous success rates.

        Analyzes the last three action success rates from metrics history and
        increases depth for high success (>0.85), decreases for low (<0.6), or
        stabilizes at 5 if consistently high (>0.8). Bounds: 3-10.

        Returns:
            None
        """
        metrics_history = self.auto_evaluator._read_metrics_history()
        actions_rates = metrics_history.get("actions_success_rate", [0.0])
        if len(actions_rates) >= 3:
            avg_success_rate = sum(actions_rates[-3:]) / 3
        else:
            avg_success_rate = actions_rates[-1] if actions_rates else 0.0

        if avg_success_rate > self.HIGH_SUCCESS_THRESHOLD:
            self.recursion_depth = min(self.MAX_RECURSION_DEPTH, self.recursion_depth + 1)
        elif avg_success_rate < self.LOW_SUCCESS_THRESHOLD:
            self.recursion_depth = max(self.MIN_RECURSION_DEPTH, self.recursion_depth - 1)

        # Stabilize at higher depth if success is consistently high
        if self.recursion_depth < self.STABLE_RECURSION_DEPTH and avg_success_rate > self.HIGH_STABLE_SUCCESS_THRESHOLD:
            self.recursion_depth = self.STABLE_RECURSION_DEPTH

    def _setup_hierarchical_planning(self, goal: str) -> None:
        """Set up hierarchical planner for persistent objectives and missions."""
        if self._hierarchical_planner:
            # Roll forward objectives to track progress
            new_subgoals = self._hierarchical_planner.roll_forward_objectives()

            # If goal matches a persistent objective, create mission
            for obj_id, objective in self._hierarchical_planner.objectives.items():
                if objective['description'] in goal and objective.get('status') == 'active':
                    mission = self._hierarchical_planner.create_mission_from_objective(obj_id)
                    if mission:
                        self.logger.info(f"Created mission {mission.id} for objective {obj_id}")

    def _apply_backlog_weights_to_subgoals(self) -> None:
        """Apply backlog weights to existing subgoals."""
        subgoals_path = self.config.workspace_root / "seed/subgoals.json"
        os.makedirs(subgoals_path.parent, exist_ok=True)
        if subgoals_path.exists():
            try:
                with open(subgoals_path, 'r') as f:
                    subgoals = json.load(f)
                # Apply weights to subgoals with high delta
                for sg in subgoals:
                    if sg.get('high_delta', False):
                        sg['priority'] = sg.get('priority', 1.0) * self.backlog_weights.get("high_delta", self.BACKLOG_HIGH_DELTA_WEIGHT)
                self.logger.info(f"Applied weights to {len(subgoals)} subgoals")
                # Save updated subgoals
                with open(subgoals_path, 'w') as f:
                    json.dump(subgoals, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to apply backlog weights: {e}")

    def _decompose_goal(self, goal: str) -> List[str]:
        """Decompose complex goal into subtasks using PlanComposer.

        Uses the PlanComposer to break down the goal into actionable subtasks.
        Saves derived subgoals to seed/subgoals.json for persistence.

        Args:
            goal (str): The original goal to decompose.

        Returns:
            List[str]: List of decomposed subtask strings. Single-item list if not complex.

        Examples:
            subtasks = agent._decompose_goal("Create a full web app")
            # Might return ["Design UI", "Implement backend", "Add tests"]
        """
        from .planning.plan_composer import PlanComposer
        plan_composer = PlanComposer()
        subtasks = plan_composer.decompose_goal(goal)

        # Save derived subgoals
        subgoals_path = self.config.workspace_root / "seed/subgoals.json"
        if len(subtasks) > 1:
            subgoals_data = [{"goal": st, "priority": 1.0, "parent_goal": goal} for st in subtasks]
            try:
                with open(subgoals_path, 'w') as f:
                    json.dump(subgoals_data, f, indent=2)
                self.logger.info(f"Saved {len(subtasks)} subgoals to {subgoals_path}")
            except Exception as e:
                self.logger.error(f"Failed to save subgoals: {e}")

        return subtasks

    def _handle_subtasks(self, subtasks: List[str], history: AgentHistory, goal: str) -> AgentResult:
        """Handles subtasks in hierarchical planning.

        Decomposes a complex goal into subtasks and executes them using a sub-agent
        approach via the PlanComposer. Aggregates results and updates history.

        Args:
            subtasks (List[str]): List of decomposed subtask goals.
            history (AgentHistory): The current execution history to append results to.
            goal (str): The parent goal from which subtasks were derived.

        Returns:
            AgentResult: Aggregated result from subtask execution, including completion
                status, iterations (set to 1 for decomposition), failures, updated history,
                errors, and estimated memories reused.

        Examples:
            subtasks = ["Analyze requirements", "Implement logic", "Write tests"]
            result = agent._handle_subtasks(subtasks, history, "Build a calculator")
            print(f"Subtasks completed: {result.completed}")
        """
        self.logger.info(f"Decomposed goal into {len(subtasks)} subtasks: {subtasks}")
        from .planning.plan_composer import PlanComposer
        plan_composer = PlanComposer()
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

    def _run_main_loop(
        self,
        goal: str,
        history: AgentHistory,
        failures: int,
        errors: List[str],
        max_iterations: int,
        context_summary: str,
        memory_lessons: str,
        metrics_history: Dict[str, List[float]],
        started_at: float,
        plan_alerts: List[str],
    ) -> AgentResult:
        """Execute the main iteration loop for a simple goal."""
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

            # Evaluate and update after action
            self._evaluate_and_update(action, observation, state, metrics_history, plan_alerts, history, failures, errors, iteration, max_iterations)

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
            errors.append("Maximum iterations reached")
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

    def _evaluate_and_update(
        self,
        action: AgentAction,
        observation: Observation,
        state: AgentState,
        metrics_history: Dict[str, List[float]],
        plan_alerts: List[str],
        history: AgentHistory,
        failures: int,
        errors: List[str],
        iteration: int,
        max_iterations: int,
    ) -> None:
        """Evaluate action result and update state, including replanning if needed."""
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
            self._seed_strategist.capture_failure(state.goal, action, observation)
            if failures > self.config.limits.max_failures:
                errors.append("Maximum failures exceeded")
                self.logger.warning(f"Max failures {self.config.limits.max_failures} reached at iteration {iteration}")
                return  # Break would be handled in caller

        if (
            action.type in {ActionType.APPLY_PATCH, ActionType.WRITE_FILE}
            and self.config.tests.auto
        ):
            self._run_auto_tests(history)

    def _gather_memory_lessons(self, goal: str) -> str:
        """Gather relevant memory lessons for the goal from semantic memory.

        Queries the SemanticMemory store for top_k relevant entries based on the goal.
        Formats snippets with titles and truncated content (max 400 chars).

        Args:
            goal (str): The current goal to query memories against.

        Returns:
            str: Formatted string of lessons or empty if memory disabled/not available.

        Examples:
            lessons = agent._gather_memory_lessons("Fix bug in executor")
            # Returns: "Useful lessons:\n1. Previous Bug Fix\nContent snippet..."
        """
        if not self.config.loop.use_memory:
            return ""

        top_k = max(0, int(self.config.loop.memory_top_k))
        if top_k <= 0:
            return ""

        if self._semantic_memory is None:
            try:
                self._semantic_memory = SemanticMemory()
            except Exception as exc:  # pragma: no cover - unexpected memory load failure
                print(f"Semantic memory unavailable: {exc}")
                self._semantic_memory = None
                return ""

        if self._semantic_memory is None:
            return ""

        try:
            results = self._semantic_memory.query(goal, top_k=top_k)
        except Exception as exc:  # pragma: no cover - memory backend errors
            print(f"Error querying semantic memory: {exc}")
            return ""

        if not results:
            return ""

        snippets: List[str] = []
        for index, (entry, _score) in enumerate(results, 1):
            content = entry.content.strip()
            if len(content) > self.MEMORY_SNIPPET_MAX_LENGTH:
                content = content[:self.MEMORY_ELLIPSIS_LENGTH] + "..."
            snippets.append(f"{index}. {entry.title}\n{content}")

        if not snippets:
            return ""

        return "Useful lessons:\n" + "\n\n".join(snippets)

    # Internos -----------------------------------------------------------------

    def _run_auto_tests(self, history: AgentHistory) -> None:
        """Run automated tests after patch or file write actions."""
        for command in self.config.tests.commands:
            action = AgentAction(type=ActionType.RUN_COMMAND, command=command)
            observation = self.executor.execute(action)
            history.append(action, observation)
            self._notify_llm(observation)

    def _notify_llm(self, observation: Observation) -> None:
        """Notify LLM of observation with truncated excerpt if necessary."""
        excerpt = observation.output
        if len(excerpt) > self.OBSERVATION_EXCERPT_MAX_LENGTH:
            excerpt = excerpt[:self.OBSERVATION_ELLIPSIS_LENGTH] + "..."
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
            print(f"Planner unavailable: {exc}")
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
            print(f"Failed to replan: {exc}")
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
            print(f"Failed to load missions.yaml: {exc}")
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
        """Record auto-evaluation metrics, derive lessons, and generate seeds.

        Computes duration, analyzes history for metrics/capabilities, updates hints
        based on success rates, calculates fitness, generates evaluation seeds for
        failures/low metrics, and persists via auto_evaluator. Also checks for new skills.

        Args:
            goal (str): The goal evaluated.
            result (AgentResult): Execution result to analyze.
            started_at (float): Perf_counter timestamp of run start.

        Returns:
            Dict[str, float]: Recorded metrics snapshot including actions, patches, tests, etc.

        Examples:
            metrics = agent._record_auto_evaluation("Test goal", result, start_time)
            print(f"Success rate: {metrics.get('actions_success_rate', 0):.2f}")
        """
        if not self.auto_evaluator:
            return {}

        duration = time.perf_counter() - started_at
        seeds: List[EvaluationSeed] = []
        metrics, inferred_caps = self._analyze_history(result)

        # Derive hints from outcomes
        success_rate = metrics.get("actions_success_rate", 0.0)
        if success_rate > self.HIGH_SUCCESS_RATE:
            current_adjust = self.hints.get("recursion_depth_adjust", 0)
            lesson = {"recursion_depth_adjust": current_adjust + 0.5}
            self.update_hints(lesson)
        elif success_rate < self.LOW_SUCCESS_RATE:
            current_adjust = self.hints.get("recursion_depth_adjust", 0)
            lesson = {"recursion_depth_adjust": max(0, current_adjust - 0.5)}
            self.update_hints(lesson)

        if "apply_patch_success_rate" in metrics:
            if metrics["apply_patch_success_rate"] > self.PATCH_HIGH_SUCCESS_RATE:
                current_bias = self.hints.get("action_biases", {}).get("APPLY_PATCH", 1.0)
                lesson = {"action_biases": {"APPLY_PATCH": current_bias + 0.1}}
                self.update_hints(lesson)
            elif metrics["apply_patch_success_rate"] < self.PATCH_LOW_SUCCESS_RATE:
                current_bias = self.hints.get("action_biases", {}).get("APPLY_PATCH", 1.0)
                lesson = {"action_biases": {"APPLY_PATCH": max(0.1, current_bias - 0.1)}}
                self.update_hints(lesson)

        # Additional learning based on other metrics
        if "test_failure_rate" in metrics and metrics["test_failure_rate"] > self.TEST_FAILURE_THRESHOLD:
            current_bias = self.hints.get("action_biases", {}).get("RUN_COMMAND", 1.0)
            lesson = {"action_biases": {"RUN_COMMAND": current_bias + 0.2}}
            self.update_hints(lesson)

        # Learning from memory usage
        if "memories_reused" in metrics and metrics["memories_reused"] > 0:
            current_memory_weight = self.hints.get("backlog_weights", {}).get("memory_usage", self.DEFAULT_BACKLOG_MEMORY_USAGE_WEIGHT)
            lesson = {"backlog_weights": {"memory_usage": current_memory_weight + 0.1}}
            self.update_hints(lesson)

        # Fitness calculation and storage
        fitness_after = self._calculate_fitness(metrics)
        fitness_before = self._get_fitness_before()
        delta = fitness_after - fitness_before
        self._save_fitness_data(fitness_before, fitness_after, delta, goal)
        metrics.update(self._aggregate_llm_metrics())

        # Generate evaluation seeds based on results
        if not result.completed:
            seeds.append(
                EvaluationSeed(
                    description="Investigate why the goal was not completed.",
                    priority="high",
                    seed_type="analysis",
                )
            )
        if result.failures:
            seeds.append(
                EvaluationSeed(
                    description="Reduce failures during execution (evaluate logs and policies).",
                    priority="medium",
                    seed_type="analysis",
                )
            )
        if metrics.get("actions_success_rate", 1.0) < 0.8:
            seeds.append(
                EvaluationSeed(
                    description="Improve action success rate (adjust prompts or policies).",
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
                    description="Investigate patch application failures and improve editing heuristics.",
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
                        description=f"Ensure automated test for metric {metric_name}.",
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

        # Register new skills if applicable
        if "skill" in goal.lower() and ("create" in goal.lower() or "implement" in goal.lower()):
            self._check_and_register_new_skills()

        return evaluation.metrics

    def _record_retrospective(
        self,
        result: AgentResult,
        metrics_snapshot: Dict[str, float],
        plan_alerts: List[str],
    ) -> None:
        """Record retrospective report and apply updates."""
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
            print(f"Failed to record retrospective: {exc}")

    def _apply_policy_overrides(self, report) -> None:
        """Apply policy overrides based on retrospective report."""
        try:
            self._policy_manager.update_from_report(report, self)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to apply policy overrides: {exc}")

    def _flush_failure_seeds(self) -> None:
        """Flush failure-derived seeds to backlog."""
        try:
            created = self._seed_strategist.flush()
            if created:
                print(f"[seeds] {len(created)} seeds added from failures")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to generate failure seeds: {exc}")

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
        """Calculate overall fitness score from weighted metrics.

        Weighted sum: actions (0.4), patches (0.3), tests (0.2), recursion (0.1, capped).
        Score between 0-1; higher indicates better performance/efficiency.

        Args:
            metrics (Dict[str, float]): Recorded metrics from execution.

        Returns:
            float: Aggregate fitness score (0.0 to 1.0).

        Examples:
            fitness = agent._calculate_fitness({"actions_success_rate": 0.9, ...})
            # Returns e.g., 0.85
        """
        fitness = 0.0

        # Actions success rate (most important)
        fitness += metrics.get("actions_success_rate", 0.0) * self.FITNESS_ACTIONS_WEIGHT

        # Apply patch success rate
        if "apply_patch_success_rate" in metrics:
            fitness += metrics["apply_patch_success_rate"] * self.FITNESS_PATCH_WEIGHT

        # Test success rate
        if "tests_success_rate" in metrics:
            fitness += metrics["tests_success_rate"] * self.FITNESS_TESTS_WEIGHT

        # Recursion depth (efficiency, capped)
        fitness += min(metrics.get("recursion_depth", 0) / self.MAX_RECURSION_FOR_FITNESS, 1.0) * self.FITNESS_RECURSION_WEIGHT

        return fitness

    def _get_fitness_before(self) -> float:
        """Retrieve fitness score from the previous run."""
        fitness_history_path = self.config.workspace_root / "seed/fitness_history.json"
        if fitness_history_path.exists():
            try:
                with open(fitness_history_path, 'r') as f:
                    history = json.load(f)
                    if history:
                        return history[-1].get('fitness_after', 0.0)
            except Exception:
                pass
        return 0.0

    def _save_fitness_data(self, fitness_before: float, fitness_after: float, delta: float, goal: str) -> None:
        """Persist fitness metrics to history file, limiting size."""
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
                pass  # Start fresh if corrupted

        # Add new entry
        entry = {
            "timestamp": time.time(),
            "goal": goal,
            "fitness_before": fitness_before,
            "fitness_after": fitness_after,
            "delta": delta,
            "recursion_depth": self.recursion_depth,
            "action_biases": self.action_biases.copy(),
            "completed": True  # Set based on result.completed in full impl
        }

        history.append(entry)

        # Limit history size
        history = history[-self.MAX_FITNESS_HISTORY_ENTRIES:]

        # Save
        try:
            with open(fitness_history_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save fitness data: {e}")

    def _check_and_register_new_skills(self) -> None:
        """Detect and register newly created skills from the run."""
        from .skills_registry import get_skills_registry
        registry = get_skills_registry(self.config.workspace_root)

        # Scan for new skill files
        for skill_file in self.config.workspace_root.glob("a3x/skills/*.py"):
            if skill_file.name.startswith("__"):
                continue

            skill_name = skill_file.stem
            if skill_name not in registry.list_skills():
                print(f"Found new skill file: {skill_file}, loading it...")
                registry.load_new_skill(str(skill_file))

                # Bias towards new skills
                try:
                    skill_class = registry.get_skill(skill_name)
                    lesson = {
                        "action_biases": {skill_name: self.NEW_SKILL_BIAS}
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
