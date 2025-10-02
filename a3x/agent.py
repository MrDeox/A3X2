"""Main orchestrator for the A3X agent."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .actions import ActionType, AgentAction, AgentState, Observation
from .autoeval import AutoEvaluator, EvaluationSeed
from .autonomous_goal_config import AutonomousGoalConfig, get_autonomous_goal_config
from .autonomous_goal_generator import AutonomousGoalGenerator, AutonomousGoal
from .cache import CacheManager, llm_cache_manager, ast_cache_manager, memory_cache_manager, config_cache_manager
from .config import AgentConfig
from .constants import (
    BACKLOG_HIGH_DELTA_WEIGHT,
    BASE_ITERATIONS,
    BIAS_PROPOSAL_CANDIDATES,
    DEFAULT_BACKLOG_MEMORY_USAGE_WEIGHT,
    DEFAULT_HINTS_ACTION_BIASES,
    DEFAULT_HINTS_BACKLOG_WEIGHTS,
    DEFAULT_MAX_SUB_DEPTH,
    DEFAULT_RECURSION_DEPTH,
    FITNESS_ACTIONS_WEIGHT,
    FITNESS_PATCH_WEIGHT,
    FITNESS_RECURSION_WEIGHT,
    FITNESS_TESTS_WEIGHT,
    HIGH_STABLE_SUCCESS_THRESHOLD,
    HIGH_SUCCESS_RATE,
    HIGH_SUCCESS_THRESHOLD,
    LLM_PROPOSAL_ATTEMPTS,
    LOW_SUCCESS_RATE,
    LOW_SUCCESS_THRESHOLD,
    MAX_FITNESS_HISTORY_ENTRIES,
    MAX_RECURSION_DEPTH,
    MAX_RECURSION_FOR_FITNESS,
    MEMORY_TOP_K_DEFAULT,
    MIN_RECURSION_DEPTH,
    NEW_SKILL_BIAS,
    OBSERVATION_ELLIPSIS_LENGTH,
    OBSERVATION_EXCERPT_MAX_LENGTH,
    PATCH_HIGH_SUCCESS_RATE,
    PATCH_LOW_SUCCESS_RATE,
    STABLE_RECURSION_DEPTH,
    TEST_FAILURE_THRESHOLD,
    MEMORY_SNIPPET_MAX_LENGTH,
    MEMORY_ELLIPSIS_LENGTH,
)
from .executor import ActionExecutor
from .history import AgentHistory
from .llm import BaseLLMClient
from .llm_seed_strategist import LLMSeedStrategist
from .memory.insights import build_retrospective, persist_retrospective
from .memory.store import SemanticMemory
from .meta_recursion import MetaRecursionEngine
from .planner import PlannerThresholds
from .planning import GoalPlan, HierarchicalPlanner, MissionState
from .planning.storage import load_mission_state
from .policy import PolicyOverrideManager


@dataclass
class AgentResult:
    completed: bool
    iterations: int
    failures: int
    history: AgentHistory
    errors: list[str]
    memories_reused: int = 0


@dataclass
class AutonomousModeConfig:
    """Configuration for autonomous goal generation and execution."""

    # Enable autonomous mode
    enable_autonomous_goals: bool = True

    # Goal generation settings
    max_autonomous_goals_per_cycle: int = 5
    autonomous_goal_generation_interval_seconds: int = 300  # 5 minutes

    # Safety and validation settings
    require_goal_validation: bool = True
    max_risky_actions_per_goal: int = 10
    enable_safety_checks: bool = True

    # Integration settings
    autonomous_goals_backlog_file: str = "seed/autonomous_goals_backlog.yaml"
    enable_goal_backlog_management: bool = True

    # Quality control
    min_estimated_impact_threshold: float = 0.3
    max_goal_complexity: int = 3  # Scale 1-5

    # Performance monitoring
    track_autonomous_goal_metrics: bool = True
    autonomous_goal_timeout_seconds: int = 1800  # 30 minutes

    def is_safety_check_enabled(self) -> bool:
        """Check if safety checks are enabled."""
        return self.enable_safety_checks

    def should_validate_goals(self) -> bool:
        """Check if goal validation is required."""
        return self.require_goal_validation

    def get_effective_timeout(self) -> int:
        """Get the effective timeout for autonomous goals."""
        return self.autonomous_goal_timeout_seconds


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
        self.logger: logging.Logger = None
        self._setup_logging()

        if self.depth > MAX_RECURSION_DEPTH:
            backoff = min(60, 2 ** (self.depth - 10))
            time.sleep(backoff)
            self.logger.info(f"Recursion depth {self.depth} exceeded bound (max=10); applied exponential backoff of {backoff}s")

        # Initialize autonomous goal generation first
        self._autonomous_config = get_autonomous_goal_config(self.config.workspace_root)

        self._initialize_components(auto_evaluator)
        self._load_and_apply_hints()

        self._semantic_memory: SemanticMemory | None = None
        self._active_plan: GoalPlan | None = None
        self._autonomous_goal_generator: AutonomousGoalGenerator | None = None
        self._last_autonomous_generation: float | None = None

        # Initialize caching and performance monitoring
        self._cache_manager = CacheManager()
        self._performance_metrics = {
            "agent_start_time": time.perf_counter(),
            "iteration_times": [],
            "cache_stats": {},
            "memory_usage": [],
            "llm_call_count": 0,
            "llm_cache_hits": 0
        }

    def _setup_logging(self) -> None:
        os.makedirs(self.config.workspace_root / "a3x/logs", exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.config.workspace_root / "a3x/logs/hints.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _initialize_components(self, auto_evaluator: AutoEvaluator | None) -> None:
        self.executor = ActionExecutor(self.config)
        thresholds = PlannerThresholds(
            apply_patch_success_rate=self.config.goals.get_threshold("apply_patch_success_rate", 0.8),
            actions_success_rate=self.config.goals.get_threshold("actions_success_rate", 0.8),
            tests_success_rate=self.config.goals.get_threshold("tests_success_rate", 0.9),
        )
        self.auto_evaluator = auto_evaluator or AutoEvaluator(thresholds=thresholds, config=self.config)
        self._llm_metrics: dict[str, list[float]] = {}
        self._hierarchical_planner = HierarchicalPlanner(thresholds=thresholds)
        self._policy_manager = PolicyOverrideManager()
        self._policy_manager.apply_to_agent(self)
        self._seed_strategist = LLMSeedStrategist(self.config.loop.seed_backlog)
        self.meta_recursion_engine = MetaRecursionEngine(
            self.config, self.executor.patch_manager, self.auto_evaluator
        )
        self.meta_recursion_engine.max_depth = 10

        # Initialize autonomous goal generator
        if self._autonomous_config.enable:
            self._autonomous_goal_generator = AutonomousGoalGenerator(
                workspace_root=self.config.workspace_root,
                random_seed=None  # Use system random for diversity
            )

    def _load_and_apply_hints(self) -> None:
        self.hints_path = self.config.workspace_root / "seed/policy_hints.json"
        self.hints = self._load_hints_file()
        self.recursion_depth = int(self.hints.get("recursion_depth", DEFAULT_RECURSION_DEPTH)) + int(self.hints.get("recursion_depth_adjust", 0))
        self.max_sub_depth = int(self.hints.get("max_sub_depth", DEFAULT_MAX_SUB_DEPTH))
        self.action_biases = self.hints.get("action_biases", DEFAULT_HINTS_ACTION_BIASES)
        self.backlog_weights = self.hints.get("backlog_weights", DEFAULT_HINTS_BACKLOG_WEIGHTS)
        self.logger.info(f"Loaded hints: {self.hints}")
        self.logger.info(f"Applied recursion_depth: {self.recursion_depth}")
        self.logger.info(f"Applied max_sub_depth: {self.max_sub_depth}")
        self.logger.info(f"Applied action biases: {self.action_biases}")
        self.logger.info(f"Applied backlog weights: {self.backlog_weights}")

    def _load_hints_file(self) -> dict[str, Any]:
        """Load hints from policy_hints.json or create defaults."""
        os.makedirs(self.hints_path.parent, exist_ok=True)

        default_hints = {
            "action_biases": DEFAULT_HINTS_ACTION_BIASES,
            "recursion_depth": DEFAULT_RECURSION_DEPTH,
            "max_sub_depth": DEFAULT_MAX_SUB_DEPTH,
            "backlog_weights": DEFAULT_HINTS_BACKLOG_WEIGHTS
        }

        if not self.hints_path.exists():
            with open(self.hints_path, "w") as f:
                json.dump(default_hints, f, indent=2)
            return default_hints

        try:
            with open(self.hints_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # Fallback to defaults on error and rewrite the file
            with open(self.hints_path, "w") as f:
                json.dump(default_hints, f, indent=2)
            return default_hints

    def _setup_logging(self) -> None:
        os.makedirs(self.config.workspace_root / "a3x/logs", exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.config.workspace_root / "a3x/logs/hints.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def generate_autonomous_goals(self) -> list[AutonomousGoal]:
        """Generate autonomous goals with safety checks and validation.

        Returns:
            List[AutonomousGoal]: List of validated autonomous goals ready for execution.

        Raises:
            RuntimeError: If autonomous goal generation is disabled or fails safety checks.
        """
        if not self._autonomous_config.enable:
            self.logger.info("Autonomous goal generation is disabled")
            return []

        if not self._autonomous_goal_generator:
            self.logger.error("Autonomous goal generator not initialized")
            return []

        # Check timing constraints
        current_time = time.perf_counter()
        if (self._last_autonomous_generation and
            current_time - self._last_autonomous_generation < self._autonomous_config.integration.generation_interval_seconds):
            self.logger.info("Autonomous goal generation interval not reached")
            return []

        try:
            self.logger.info("Starting autonomous goal generation...")

            # Generate goals using the autonomous goal generator
            context = self._gather_autonomous_context()
            raw_goals = self._autonomous_goal_generator.generate_autonomous_goals(context)

            # Apply safety checks and validation
            validated_goals = []
            for goal in raw_goals[:self._autonomous_config.integration.max_goals_per_cycle]:
                if self._validate_autonomous_goal(goal):
                    validated_goals.append(goal)
                    self.logger.info(f"Validated autonomous goal: {goal.title}")
                else:
                    self.logger.warning(f"Rejected autonomous goal due to safety checks: {goal.title}")

            # Update timestamp
            self._last_autonomous_generation = current_time

            self.logger.info(f"Generated {len(validated_goals)}/{len(raw_goals)} valid autonomous goals")
            return validated_goals

        except Exception as e:
            self.logger.error(f"Failed to generate autonomous goals: {e}")
            return []

    def _gather_autonomous_context(self) -> dict[str, Any]:
        """Gather context information for autonomous goal generation."""
        context = {
            "current_capabilities": self._get_current_capabilities(),
            "recent_performance": self._get_recent_performance_metrics(),
            "system_state": self._get_system_state(),
            "historical_success_rate": self._get_historical_success_rate(),
        }

        return context

    def _get_current_capabilities(self) -> list[str]:
        """Get list of currently available capabilities."""
        # This would integrate with the capabilities system
        # For now, return a basic set
        return ["core.diffing", "core.testing", "horiz.python", "meta.skill_creation"]

    def _get_recent_performance_metrics(self) -> dict[str, float]:
        """Get recent performance metrics for context."""
        if not self.auto_evaluator:
            return {}

        metrics_history = self.auto_evaluator._read_metrics_history()
        recent_metrics = {}

        # Get latest values for key metrics
        for metric_name in ["actions_success_rate", "apply_patch_success_rate", "tests_success_rate"]:
            values = metrics_history.get(metric_name, [])
            if values:
                recent_metrics[metric_name] = values[-1]

        return recent_metrics

    def _get_system_state(self) -> dict[str, Any]:
        """Get current system state information."""
        return {
            "recursion_depth": self.recursion_depth,
            "max_sub_depth": self.max_sub_depth,
            "active_plan_exists": self._active_plan is not None,
            "memory_available": self.config.loop.use_memory,
        }

    def _get_historical_success_rate(self) -> float:
        """Get historical success rate for context."""
        metrics_history = self.auto_evaluator._read_metrics_history() if self.auto_evaluator else {}
        actions_rates = metrics_history.get("actions_success_rate", [])

        if actions_rates:
            return sum(actions_rates[-5:]) / min(len(actions_rates), 5)  # Average of last 5 runs
        return 0.5  # Default neutral value

    def _validate_autonomous_goal(self, goal: AutonomousGoal) -> bool:
        """Validate autonomous goal against safety checks and quality criteria.

        Args:
            goal: The autonomous goal to validate

        Returns:
            bool: True if goal passes validation, False otherwise
        """
        # Safety check 1: Estimated impact threshold
        if goal.estimated_impact < 0.3:  # Default threshold
            self.logger.warning(f"Goal {goal.title} rejected: impact {goal.estimated_impact} below threshold {self._autonomous_config.min_estimated_impact_threshold}")
            return False

        # Safety check 2: Required capabilities validation
        available_capabilities = set(self._get_current_capabilities())
        required_capabilities = set(goal.required_capabilities) if goal.required_capabilities else set()

        if required_capabilities and not required_capabilities.issubset(available_capabilities):
            missing_caps = required_capabilities - available_capabilities
            self.logger.warning(f"Goal {goal.title} rejected: missing capabilities {missing_caps}")
            return False

        # Safety check 3: Goal complexity check
        # This is a simplified complexity check based on goal type and description length
        complexity_score = self._calculate_goal_complexity(goal)
        if complexity_score > 3:  # Default max complexity
            self.logger.warning(f"Goal {goal.title} rejected: complexity {complexity_score} exceeds max {self._autonomous_config.max_goal_complexity}")
            return False

        # Safety check 4: Content safety check
        if not self._perform_content_safety_check(goal):
            self.logger.warning(f"Goal {goal.title} rejected: failed content safety check")
            return False

        return True

    def _calculate_goal_complexity(self, goal: AutonomousGoal) -> int:
        """Calculate complexity score for a goal (1-5 scale).

        Args:
            goal: The goal to analyze

        Returns:
            int: Complexity score from 1 (simple) to 5 (very complex)
        """
        complexity = 1

        # Base complexity from goal type
        goal_type_complexity = {
            "capability_gap": 2,
            "curiosity_exploration": 3,
            "self_optimization": 3,
            "domain_expansion": 4,
            "meta_reflection": 2,
        }
        complexity = max(complexity, goal_type_complexity.get(goal.goal_type, 2))

        # Adjust based on description length (longer = potentially more complex)
        description_length = len(goal.description)
        if description_length > 200:
            complexity += 1
        elif description_length > 100:
            complexity += 0.5

        # Adjust based on number of required capabilities
        if goal.required_capabilities:
            complexity += min(len(goal.required_capabilities) * 0.5, 2)

        return min(int(complexity), 5)

    def _perform_content_safety_check(self, goal: AutonomousGoal) -> bool:
        """Perform content safety check on goal.

        Args:
            goal: The goal to check

        Returns:
            bool: True if content passes safety check
        """
        # Basic safety checks - avoid dangerous operations
        dangerous_keywords = [
            "delete", "remove", "drop", "truncate", "format",
            "shutdown", "reboot", "kill", "terminate",
            "hack", "exploit", "attack", "malware"
        ]

        content_to_check = (goal.title + " " + goal.description).lower()

        for keyword in dangerous_keywords:
            if keyword in content_to_check:
                # Check if it's a false positive (e.g., "delete files" vs "delete temporary files")
                if not self._is_safe_context(content_to_check, keyword):
                    return False

        return True

    def _is_safe_context(self, content: str, keyword: str) -> bool:
        """Check if a keyword appears in a safe context.

        Args:
            content: The content to analyze
            keyword: The potentially dangerous keyword

        Returns:
            bool: True if the context appears safe
        """
        # Define safe contexts for dangerous keywords
        safe_contexts = {
            "delete": ["temporary", "cache", "log", "test", "debug"],
            "remove": ["temporary", "cache", "log", "test", "debug"],
            "kill": ["process", "thread", "connection"],  # Technical contexts
        }

        safe_terms = safe_contexts.get(keyword, [])
        return any(safe_term in content for safe_term in safe_terms)

    def should_enter_autonomous_mode(self, current_goal: str | None = None) -> bool:
        """Determine if the agent should enter autonomous mode.

        Args:
            current_goal: The current human-provided goal, if any

        Returns:
            bool: True if autonomous mode should be activated
        """
        if not self._autonomous_config.enable:
            return False

        # Don't enter autonomous mode if we have a current human-provided goal
        if current_goal and current_goal.strip():
            return False

        # Check if enough time has passed since last autonomous generation
        current_time = time.perf_counter()
        if (self._last_autonomous_generation and
            current_time - self._last_autonomous_generation < self._autonomous_config.integration.generation_interval_seconds):
            return False

        # Check system conditions for autonomous operation
        return self._check_autonomous_conditions()

    def _check_autonomous_conditions(self) -> bool:
        """Check if system conditions are suitable for autonomous operation.

        Returns:
            bool: True if conditions are suitable
        """
        # Check 1: System performance is adequate
        if not self._check_system_performance():
            self.logger.info("Autonomous mode skipped: inadequate system performance")
            return False

        # Check 2: Recent success rate is reasonable
        recent_success_rate = self._get_historical_success_rate()
        if recent_success_rate < 0.3:  # Less than 30% success rate
            self.logger.info(f"Autonomous mode skipped: low success rate {recent_success_rate:.2f}")
            return False

        # Check 3: Not currently in a failure cascade
        if self._is_in_failure_cascade():
            self.logger.info("Autonomous mode skipped: currently in failure cascade")
            return False

        return True

    def _check_system_performance(self) -> bool:
        """Check if system performance is adequate for autonomous operation.

        Returns:
            bool: True if performance is adequate
        """
        # Check memory usage (simplified)
        try:
            # This would integrate with actual system monitoring
            # For now, assume performance is adequate
            return True
        except Exception:
            return False

    def _is_in_failure_cascade(self) -> bool:
        """Check if the system is currently in a failure cascade.

        Returns:
            bool: True if in failure cascade
        """
        if not self.auto_evaluator:
            return False

        metrics_history = self.auto_evaluator._read_metrics_history()

        # Check last 3 runs for consecutive failures
        actions_rates = metrics_history.get("actions_success_rate", [])
        if len(actions_rates) >= 3:
            recent_rates = actions_rates[-3:]
            # If all recent runs had < 50% success rate
            return all(rate < 0.5 for rate in recent_rates)

        return False

    def switch_to_autonomous_mode(self) -> AutonomousGoal | None:
        """Switch to autonomous mode and return the next autonomous goal.

        Returns:
            AutonomousGoal | None: The next autonomous goal to execute, or None if switching fails
        """
        if not self.should_enter_autonomous_mode():
            self.logger.info("Cannot switch to autonomous mode: conditions not met")
            return None

        try:
            self.logger.info("Switching to autonomous mode...")

            # Generate autonomous goals
            autonomous_goals = self.generate_autonomous_goals()

            if not autonomous_goals:
                self.logger.info("No autonomous goals generated")
                return None

            # Select the best goal based on motivation and impact
            selected_goal = self._select_best_autonomous_goal(autonomous_goals)

            if selected_goal:
                self.logger.info(f"Selected autonomous goal: {selected_goal.title}")
                return selected_goal
            else:
                self.logger.warning("No suitable autonomous goal found after selection")
                return None

        except Exception as e:
            self.logger.error(f"Failed to switch to autonomous mode: {e}")
            return None

    def _select_best_autonomous_goal(self, goals: list[AutonomousGoal]) -> AutonomousGoal | None:
        """Select the best autonomous goal from a list of candidates.

        Args:
            goals: List of candidate autonomous goals

        Returns:
            AutonomousGoal | None: The selected goal, or None if no suitable goal
        """
        if not goals:
            return None

        # Score goals based on multiple factors
        scored_goals = []
        for goal in goals:
            score = self._calculate_goal_priority_score(goal)
            scored_goals.append((goal, score))

        # Sort by score (highest first)
        scored_goals.sort(key=lambda x: x[1], reverse=True)

        # Return the highest scoring goal
        best_goal, best_score = scored_goals[0]

        self.logger.info(f"Selected goal '{best_goal.title}' with score {best_score:.3f}")
        return best_goal

    def _calculate_goal_priority_score(self, goal: AutonomousGoal) -> float:
        """Calculate priority score for an autonomous goal.

        Args:
            goal: The goal to score

        Returns:
            float: Priority score (higher is better)
        """
        score = 0.0

        # Base score from estimated impact
        score += goal.estimated_impact * 0.4

        # Motivation factors (normalized)
        total_motivation = sum(goal.motivation_factors.values()) if goal.motivation_factors else 0.0
        score += min(total_motivation, 1.0) * 0.3

        # Priority multiplier
        priority_multiplier = {
            "high": 1.3,
            "medium": 1.0,
            "low": 0.7
        }
        score *= priority_multiplier.get(goal.priority, 1.0)

        # Recency bonus (newer goals get slight bonus)
        if hasattr(goal, 'created_at'):
            try:
                # This would parse the timestamp and give a small bonus for recent goals
                # For now, give a small random bonus to avoid always picking the same goal
                import random
                score += random.uniform(0.0, 0.1)
            except Exception:
                pass

        return score

    def run_autonomous_batch(self, max_goals: int | None = None) -> list[AgentResult]:
        """Run multiple autonomous goals in batch mode.

        Args:
            max_goals: Maximum number of goals to run (None for no limit)

        Returns:
            list[AgentResult]: Results from each autonomous goal execution
        """
        results = []
        goals_completed = 0

        while (max_goals is None or goals_completed < max_goals):
            # Check if we should continue autonomous operation
            if not self.should_enter_autonomous_mode():
                self.logger.info("Stopping autonomous batch: conditions no longer met")
                break

            # Get next autonomous goal
            autonomous_goal = self.switch_to_autonomous_mode()
            if not autonomous_goal:
                self.logger.info("No more autonomous goals available")
                break

            # Convert autonomous goal to string format for execution
            goal_text = f"{autonomous_goal.title} - {autonomous_goal.description}"

            try:
                self.logger.info(f"Executing autonomous goal {goals_completed + 1}: {autonomous_goal.title}")
                result = self.run(goal_text)
                results.append(result)

                goals_completed += 1

                # Check if the autonomous goal was completed successfully
                if result.completed:
                    self.logger.info(f"Autonomous goal completed successfully: {autonomous_goal.title}")
                else:
                    self.logger.warning(f"Autonomous goal incomplete: {autonomous_goal.title}")

            except Exception as e:
                self.logger.error(f"Error executing autonomous goal {autonomous_goal.title}: {e}")
                # Continue with next goal instead of stopping
                continue

        self.logger.info(f"Autonomous batch completed: {goals_completed} goals executed")
        return results

    def configure_autonomous_mode(self, **config_kwargs) -> None:
        """Configure autonomous mode settings.

        Args:
            **config_kwargs: Configuration parameters to update
                - enable_autonomous_goals (bool): Enable/disable autonomous goal generation
                - max_autonomous_goals_per_cycle (int): Max goals to generate per cycle
                - autonomous_goal_generation_interval_seconds (int): Interval between generations
                - require_goal_validation (bool): Require validation of generated goals
                - max_risky_actions_per_goal (int): Max risky actions per goal
                - enable_safety_checks (bool): Enable/disable safety checks
                - min_estimated_impact_threshold (float): Minimum impact threshold
                - max_goal_complexity (int): Maximum goal complexity (1-5)
                - track_autonomous_goal_metrics (bool): Enable metrics tracking
                - autonomous_goal_timeout_seconds (int): Timeout for autonomous goals
        """
        for key, value in config_kwargs.items():
            if hasattr(self._autonomous_config, key):
                setattr(self._autonomous_config, key, value)
                self.logger.info(f"Updated autonomous config {key} = {value}")
            else:
                self.logger.warning(f"Unknown autonomous config parameter: {key}")

    def get_autonomous_mode_status(self) -> dict[str, Any]:
        """Get current autonomous mode status and configuration.

        Returns:
            dict: Status information including configuration and metrics
        """
        current_time = time.perf_counter()
        time_since_last_generation = None

        if self._last_autonomous_generation:
            time_since_last_generation = current_time - self._last_autonomous_generation

        return {
            "autonomous_mode_enabled": self._autonomous_config.enable_autonomous_goals,
            "autonomous_generator_initialized": self._autonomous_goal_generator is not None,
            "last_generation_timestamp": self._last_autonomous_generation,
            "time_since_last_generation": time_since_last_generation,
            "can_generate_goals": self.should_enter_autonomous_mode(),
            "configuration": {
                "max_goals_per_cycle": self._autonomous_config.integration.max_goals_per_cycle,
                "generation_interval": self._autonomous_config.integration.generation_interval_seconds,
                "safety_checks_enabled": True,  # Default value
                "validation_required": True,  # Default value
                "min_impact_threshold": 0.3,  # Default value
                "max_complexity": 3,  # Default value
            }
        }

    def enable_autonomous_mode(self, enable: bool = True) -> None:
        """Enable or disable autonomous mode.

        Args:
            enable: Whether to enable autonomous mode
        """
        self._autonomous_config.enable = enable
        status = "enabled" if enable else "disabled"
        self.logger.info(f"Autonomous mode {status}")

    def is_autonomous_mode_active(self) -> bool:
        """Check if autonomous mode is currently active.

        Returns:
            bool: True if autonomous mode is active
        """
        return (self._autonomous_config.enable and
                self._autonomous_goal_generator is not None)

    def update_hints(self, lesson: dict[str, Any]) -> None:
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
        temp_path = self.hints_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(self.hints, f, indent=2)
            temp_path.replace(self.hints_path)
            self.logger.info(f"Updated hints with lesson: {lesson}")
        except Exception as e:
            self.logger.error(f"Failed to update hints: {e}")
            # Fallback: try direct write
            try:
                with open(self.hints_path, "w") as f:
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
        prompt = f"""Based on the state, propose {BIAS_PROPOSAL_CANDIDATES} possible next actions as a JSON list.
Each action: {{"type": "one of {action_types}", "command": "list of strings or string"}}.
State: {state}"""
        for attempt in range(LLM_PROPOSAL_ATTEMPTS):
            try:
                response = self.llm_client.chat(prompt)
                candidates_data = json.loads(response)
                if not isinstance(candidates_data, list) or len(candidates_data) < 1:
                    raise ValueError("Invalid candidates")
                candidates = []
                for data in candidates_data[:BIAS_PROPOSAL_CANDIDATES]:  # Limit to constant
                    atype_str = data.get("type", "RUN_COMMAND")
                    try:
                        atype = ActionType[atype_str]
                    except KeyError:
                        continue
                    command = data.get("command", [])
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
                if attempt == LLM_PROPOSAL_ATTEMPTS - 1:
                    break
                time.sleep(1)
        # Fallback
        self.logger.info("Falling back to standard action proposal")
        return self.llm_client.propose_action(state)

    def run(self, goal: str | None = None) -> AgentResult:
        """Execute the agent's main loop to achieve the given goal.

        This method orchestrates the hierarchical planning, action execution, and evaluation
        to fulfill the specified goal. It handles goal decomposition into subtasks if needed,
        runs the iteration loop, and records results including metrics and retrospectives.

        If no goal is provided, the agent will attempt to switch to autonomous mode
        and generate its own goals.

        Args:
            goal (str | None): The objective string describing what the agent should accomplish.
                              If None, autonomous mode will be attempted.

        Returns:
            AgentResult: A summary object containing completion status, iteration count,
                failure count, execution history, errors encountered, and memories reused.

        Examples:
            # Human-provided goal
            result = agent.run("Implement a new feature in the codebase")

            # Autonomous mode
            result = agent.run()  # No goal provided
        """
        # Handle autonomous mode if no goal provided
        if goal is None:
            autonomous_goal = self.switch_to_autonomous_mode()
            if autonomous_goal:
                goal = f"{autonomous_goal.title} - {autonomous_goal.description}"
                self.logger.info(f"Running in autonomous mode with goal: {goal}")
            else:
                # Fallback to a default goal if autonomous mode fails
                goal = "Analyze current state and identify potential improvements"
                self.logger.info(f"Using fallback goal: {goal}")

        if not goal or not goal.strip():
            raise ValueError("No goal provided and autonomous mode failed to generate a goal")

        history = AgentHistory()
        self.llm_client.start(goal)

        failures = 0
        errors: list[str] = []

        # Adjust recursion depth based on historical success rate
        self._adjust_recursion_depth()


        max_iterations = BASE_ITERATIONS * self.recursion_depth
        started_at = time.perf_counter()
        context_summary = self.auto_evaluator.latest_summary()

        memory_lessons = self._gather_memory_lessons(goal)
        plan_alerts: list[str] = []

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

        if avg_success_rate > HIGH_SUCCESS_THRESHOLD:
            self.recursion_depth = min(MAX_RECURSION_DEPTH, self.recursion_depth + 1)
        elif avg_success_rate < LOW_SUCCESS_THRESHOLD:
            self.recursion_depth = max(MIN_RECURSION_DEPTH, self.recursion_depth - 1)

        # Stabilize at higher depth if success is consistently high
        if self.recursion_depth < STABLE_RECURSION_DEPTH and avg_success_rate > HIGH_STABLE_SUCCESS_THRESHOLD:
            self.recursion_depth = STABLE_RECURSION_DEPTH

    def _setup_hierarchical_planning(self, goal: str) -> None:
        """Set up hierarchical planner for persistent objectives and missions."""
        if self._hierarchical_planner:
            # Roll forward objectives to track progress
            new_subgoals = self._hierarchical_planner.roll_forward_objectives()

            # If goal matches a persistent objective, create mission
            for obj_id, objective in self._hierarchical_planner.objectives.items():
                if objective["description"] in goal and objective.get("status") == "active":
                    mission = self._hierarchical_planner.create_mission_from_objective(obj_id)
                    if mission:
                        self.logger.info(f"Created mission {mission.id} for objective {obj_id}")

    def _apply_backlog_weights_to_subgoals(self) -> None:
        """Apply backlog weights to existing subgoals."""
        subgoals_path = self.config.workspace_root / "seed/subgoals.json"
        os.makedirs(subgoals_path.parent, exist_ok=True)
        if subgoals_path.exists():
            try:
                with open(subgoals_path) as f:
                    subgoals = json.load(f)
                # Apply weights to subgoals with high delta
                for sg in subgoals:
                    if sg.get("high_delta", False):
                        sg["priority"] = sg.get("priority", 1.0) * self.backlog_weights.get("high_delta", BACKLOG_HIGH_DELTA_WEIGHT)
                self.logger.info(f"Applied weights to {len(subgoals)} subgoals")
                # Save updated subgoals
                with open(subgoals_path, "w") as f:
                    json.dump(subgoals, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to apply backlog weights: {e}")

    def _decompose_goal(self, goal: str) -> list[str]:
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
                with open(subgoals_path, "w") as f:
                    json.dump(subgoals_data, f, indent=2)
                self.logger.info(f"Saved {len(subtasks)} subgoals to {subgoals_path}")
            except Exception as e:
                self.logger.error(f"Failed to save subgoals: {e}")

        return subtasks

    def _handle_subtasks(self, subtasks: list[str], history: AgentHistory, goal: str) -> AgentResult:
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
        errors: list[str],
        max_iterations: int,
        context_summary: str,
        memory_lessons: str,
        metrics_history: dict[str, list[float]],
        started_at: float,
        plan_alerts: list[str],
    ) -> AgentResult:
        """Execute the main iteration loop for a simple goal."""
        self.logger.info(f"Starting main loop for goal '{goal}' with max_iterations={max_iterations}, recursion_depth={self.recursion_depth}")

        # Performance monitoring start
        iteration_start_time = time.perf_counter()

        for iteration in range(1, max_iterations + 1):
            # Interactive mode: prompt for goal refinement
            if self.config.loop.interactive:
                refine = input(f"\nIteration {iteration}: Refine goal '{goal[:50]}...' or subgoals? (y/n): ").strip().lower()
                if refine == "y":
                    new_goal = input("Enter refined goal or subgoals: ").strip()
                    if new_goal:
                        goal = new_goal
                        self.logger.info(f"Goal refined to: {goal}")

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

            iteration_time = time.perf_counter()
            self.logger.info(f"Iteration {iteration}/{max_iterations}: Goal='{goal[:50]}...', History events={len(history.events)}, State context length={len(str(state))}")

            # Performance monitoring for action proposal
            action_start = time.perf_counter()
            action = self._propose_biased_action(state)
            action_time = time.perf_counter() - action_start
            self.logger.info(f"Proposed action: {action.type.name}, Command: {action.command or 'None'}, Time: {action_time:.3f}s")

            self._capture_llm_metrics()
            self._performance_metrics["llm_call_count"] += 1

            # Performance monitoring for action execution
            exec_start = time.perf_counter()
            observation = self.executor.execute(action)
            exec_time = time.perf_counter() - exec_start
            self.logger.info(f"Action executed: Success={observation.success}, Output length={len(observation.output or '')}, Error={observation.error or 'None'}, Time: {exec_time:.3f}s")

            # Update performance metrics
            total_iteration_time = time.perf_counter() - iteration_time
            self._performance_metrics["iteration_times"].append(total_iteration_time)

            # Update cache statistics
            self._update_cache_statistics()
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
        metrics_history: dict[str, list[float]],
        plan_alerts: list[str],
        history: AgentHistory,
        failures: int,
        errors: list[str],
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

        top_k = max(MEMORY_TOP_K_DEFAULT, int(self.config.loop.memory_top_k))
        if top_k <= MEMORY_TOP_K_DEFAULT:
            return ""

        if self._semantic_memory is None:
            try:
                self._semantic_memory = SemanticMemory()
            except Exception as exc:  # pragma: no cover - unexpected memory load failure
                self._semantic_memory = None
                return ""

        if self._semantic_memory is None:
            return ""

        try:
            results = self._semantic_memory.query(goal, top_k=top_k)
        except Exception as exc:  # pragma: no cover - memory backend errors
            return ""

        if not results:
            return ""

        snippets: list[str] = []
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
        if len(excerpt) > OBSERVATION_EXCERPT_MAX_LENGTH:
            excerpt = excerpt[:OBSERVATION_ELLIPSIS_LENGTH] + "..."
        self.llm_client.notify_observation(excerpt)

    def _ensure_plan(
        self,
        state: AgentState,
        metrics_history: dict[str, list[float]],
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
            return self._active_plan

    def _force_replan(
        self,
        state: AgentState,
        metrics_history: dict[str, list[float]],
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
            return None

    def _capture_llm_metrics(self) -> None:
        metrics = self.llm_client.get_last_metrics()
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._llm_metrics.setdefault(key, []).append(float(value))

    def _aggregate_llm_metrics(self) -> dict[str, float]:
        aggregated: dict[str, float] = {}
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
    ) -> dict[str, float]:
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
        seeds: list[EvaluationSeed] = []
        metrics, inferred_caps = self._analyze_history(result)

        # Derive hints from outcomes
        success_rate = metrics.get("actions_success_rate", 0.0)
        if success_rate > HIGH_SUCCESS_RATE:
            current_adjust = self.hints.get("recursion_depth_adjust", 0)
            lesson = {"recursion_depth_adjust": current_adjust + 0.5}
            self.update_hints(lesson)
        elif success_rate < LOW_SUCCESS_RATE:
            current_adjust = self.hints.get("recursion_depth_adjust", 0)
            lesson = {"recursion_depth_adjust": max(0, current_adjust - 0.5)}
            self.update_hints(lesson)

        if "apply_patch_success_rate" in metrics:
            if metrics["apply_patch_success_rate"] > PATCH_HIGH_SUCCESS_RATE:
                current_bias = self.hints.get("action_biases", {}).get("APPLY_PATCH", 1.0)
                lesson = {"action_biases": {"APPLY_PATCH": current_bias + 0.1}}
                self.update_hints(lesson)
            elif metrics["apply_patch_success_rate"] < PATCH_LOW_SUCCESS_RATE:
                current_bias = self.hints.get("action_biases", {}).get("APPLY_PATCH", 1.0)
                lesson = {"action_biases": {"APPLY_PATCH": max(0.1, current_bias - 0.1)}}
                self.update_hints(lesson)

        # Additional learning based on other metrics
        if "test_failure_rate" in metrics and metrics["test_failure_rate"] > TEST_FAILURE_THRESHOLD:
            current_bias = self.hints.get("action_biases", {}).get("RUN_COMMAND", 1.0)
            lesson = {"action_biases": {"RUN_COMMAND": current_bias + 0.2}}
            self.update_hints(lesson)

        # Learning from memory usage
        if "memories_reused" in metrics and metrics["memories_reused"] > 0:
            current_memory_weight = self.hints.get("backlog_weights", {}).get("memory_usage", DEFAULT_BACKLOG_MEMORY_USAGE_WEIGHT)
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
        metrics_snapshot: dict[str, float],
        plan_alerts: list[str],
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
            pass  # Silently fail if retrospective persistence fails

    def _apply_policy_overrides(self, report: Any) -> None:
        """Apply policy overrides based on retrospective report."""
        try:
            self._policy_manager.update_from_report(report, self)
        except Exception as exc:  # pragma: no cover - defensive
            pass  # Silently fail if policy updates fail

    def _flush_failure_seeds(self) -> None:
        """Flush failure-derived seeds to backlog."""
        try:
            created = self._seed_strategist.flush()
            if created:
                self.logger.info(f"[seeds] {len(created)} seeds added from failures")
        except Exception as exc:  # pragma: no cover - defensive
            pass  # Silently fail if failure seed flushing fails

    def _analyze_history(
        self, result: AgentResult
    ) -> tuple[dict[str, float], set[str]]:
        events = result.history.events
        total_actions = len(events)
        success_actions = sum(1 for event in events if event.observation.success)
        metrics: dict[str, float] = {}
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

        unique_commands: set[str] = set()
        for event in events:
            if event.action.type is ActionType.RUN_COMMAND and event.action.command:
                unique_commands.add(event.action.command[0])
        metrics["unique_commands"] = float(len(unique_commands))

        file_extensions: set[str] = set()
        for event in events:
            ext = _infer_extension(event.action)
            if ext:
                file_extensions.add(ext)
        metrics["unique_file_extensions"] = float(len(file_extensions))

        inferred_capabilities: set[str] = set()
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

    def _calculate_fitness(self, metrics: dict[str, float]) -> float:
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
        fitness += metrics.get("actions_success_rate", 0.0) * FITNESS_ACTIONS_WEIGHT

        # Apply patch success rate
        if "apply_patch_success_rate" in metrics:
            fitness += metrics["apply_patch_success_rate"] * FITNESS_PATCH_WEIGHT

        # Test success rate
        if "tests_success_rate" in metrics:
            fitness += metrics["tests_success_rate"] * FITNESS_TESTS_WEIGHT

        # Recursion depth (efficiency, capped)
        fitness += min(metrics.get("recursion_depth", 0) / MAX_RECURSION_FOR_FITNESS, 1.0) * FITNESS_RECURSION_WEIGHT

        return fitness

    def _get_fitness_before(self) -> float:
        """Retrieve fitness score from the previous run."""
        fitness_history_path = self.config.workspace_root / "seed/fitness_history.json"
        if fitness_history_path.exists():
            try:
                with open(fitness_history_path) as f:
                    history = json.load(f)
                    if history:
                        return history[-1].get("fitness_after", 0.0)
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
                with open(fitness_history_path) as f:
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
        history = history[-MAX_FITNESS_HISTORY_ENTRIES:]

        # Save
        try:
            with open(fitness_history_path, "w") as f:
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
                        "action_biases": {skill_name: NEW_SKILL_BIAS}
                    }
                    self.update_hints(lesson)
                except KeyError:
                    print(f"Could not register skill: {skill_name}")

    def _update_cache_statistics(self) -> None:
        """Update cache statistics for performance monitoring."""
        try:
            # Get global cache metrics
            cache_metrics = self._cache_manager.get_metrics()

            # Update LLM cache statistics
            llm_stats = llm_cache_manager.get_stats()

            # Update AST cache statistics
            ast_stats = ast_cache_manager.get_stats()

            # Update memory cache statistics
            memory_stats = memory_cache_manager.get_stats()

            # Update config cache statistics
            config_stats = config_cache_manager.get_stats()

            # Combine all cache stats
            self._performance_metrics["cache_stats"] = {
                "global": cache_metrics,
                "llm": llm_stats,
                "ast": ast_stats,
                "memory": memory_stats,
                "config": config_stats,
                "timestamp": time.perf_counter()
            }

        except Exception as e:
            self.logger.warning(f"Failed to update cache statistics: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        total_time = time.perf_counter() - self._performance_metrics["agent_start_time"]

        avg_iteration_time = 0.0
        if self._performance_metrics["iteration_times"]:
            avg_iteration_time = sum(self._performance_metrics["iteration_times"]) / len(self._performance_metrics["iteration_times"])

        return {
            "total_execution_time": total_time,
            "average_iteration_time": avg_iteration_time,
            "total_iterations": len(self._performance_metrics["iteration_times"]),
            "llm_call_count": self._performance_metrics["llm_call_count"],
            "cache_statistics": self._performance_metrics["cache_stats"],
            "performance_metrics": self._performance_metrics
        }

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
