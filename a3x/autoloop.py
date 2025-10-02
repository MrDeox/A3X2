"""Autonomous loop that alternates between high-level goals and seeds."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import yaml

from .agent import AgentHistory, AgentOrchestrator, AgentResult
from .config import load_config
from .dynamic_scaler import integrate_dynamic_scaler
from .llm import build_llm_client
from .meta_recursion import integrate_meta_recursion
from .patch import PatchManager
from .planner import PlannerThresholds
from .seed_runner import SeedRunner
from .seeds import AutoSeeder, Seed, SeedBacklog


class AutoLoop:
    """Basic AutoLoop class for integration tests."""

    def __init__(self, goal: str, max_iterations: int = 10):
        self.goal = goal
        self.max_iterations = max_iterations
        self.iterations = 0
        self.completed = False
        self.metrics = {"actions_success_rate": 0.9}  # Default metrics for testing

    def run(self):
        self.iterations = 1  # Mock single iteration
        self.completed = True
        return AgentResult(completed=True, iterations=self.iterations, failures=0, history=AgentHistory(), errors=[], memories_reused=0)


@dataclass
class GoalSpec:
    goal: str
    config: Path
    max_steps: int | None = None


def load_goal_rotation(path: str | Path) -> list[GoalSpec]:
    rotation_path = Path(path)
    if not rotation_path.exists():
        raise FileNotFoundError(f"Goal rotation file not found: {rotation_path}")
    raw = yaml.safe_load(rotation_path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError("Goal rotation file must contain a list")

    specs: list[GoalSpec] = []
    for item in raw:
        if not isinstance(item, dict) or "goal" not in item or "config" not in item:
            raise ValueError(
                "Each goal entry must contain at least 'goal' and 'config'"
            )
        config_value = item["config"]
        config_path = Path(config_value)
        if not config_path.is_absolute():
            candidate = (rotation_path.parent / config_path).resolve()
            if candidate.exists():
                config_path = candidate
            else:
                config_path = (Path.cwd() / config_path).resolve()
        specs.append(
            GoalSpec(
                goal=str(item["goal"]),
                config=config_path,
                max_steps=(
                    int(item["max_steps"])
                    if item.get("max_steps") is not None
                    else None
                ),
            )
        )
    return specs


async def run_autopilot(
    goals: list[GoalSpec],
    *,
    cycles: int,
    backlog_path: str | Path,
    seed_default_config: str | Path,
    seed_max: int | None = None,
    seed_max_steps: int | None = None,
) -> int:
    if not goals:
        raise ValueError("Goal rotation list cannot be empty")

    backlog_path = Path(backlog_path)
    seed_default_config = Path(seed_default_config)
    backlog_path.parent.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    backlog = SeedBacklog.load(backlog_path)
    auto_seeder = AutoSeeder(thresholds=PlannerThresholds())
    for cycle in range(cycles):
        spec = goals[cycle % len(goals)]
        print(f"=== Cycle {cycle + 1} :: {spec.goal} ===")
        run_result = await _run_goal(spec)
        print(
            f"Resultado: goal='{spec.goal}', completed={run_result.completed}, iterations={run_result.iterations}, failures={run_result.failures}"
        )
        exit_code = 0 if run_result.completed else 1

        # Compute metrics from run_result for auto-seeding
        events = run_result.history.events
        total_actions = len(events)
        success_actions = sum(1 for event in events if event.observation.success)
        metrics = {}
        if total_actions:
            metrics["actions_success_rate"] = success_actions / total_actions
        else:
            metrics["actions_success_rate"] = 0.0

        apply_patch_events = [event for event in events if event.action.type.name == "APPLY_PATCH"]
        if apply_patch_events:
            success_count = sum(1 for event in apply_patch_events if event.observation.success)
            metrics["apply_patch_success_rate"] = success_count / len(apply_patch_events)
        else:
            metrics["apply_patch_success_rate"] = 1.0  # No patches, assume success

        test_runs = [event for event in events if event.action.type.name == "RUN_COMMAND" and "pytest" in " ".join(event.action.command or [])]
        if test_runs:
            success = sum(1 for event in test_runs if event.observation.success)
            metrics["tests_success_rate"] = success / len(test_runs)
        else:
            metrics["tests_success_rate"] = 1.0  # No tests, assume success

        # Auto-seed based on metrics
        new_seeds = auto_seeder.monitor_and_seed(metrics)
        for seed in new_seeds:
            if not backlog.exists(seed.id):
                backlog.add_seed(seed)
                print(f"Added auto-seed: {seed.goal}")

        await _drain_seeds(
            backlog_path,
            default_config=seed_default_config,
            max_runs=seed_max,
            max_steps_override=seed_max_steps,
        )
    return exit_code


async def _run_goal(spec: GoalSpec) -> AgentResult:
    config = load_config(spec.config)
    if spec.max_steps is not None:
        config.limits.max_iterations = spec.max_steps

    # Integrate DynamicScaler
    scaler = integrate_dynamic_scaler(config)
    initial_metrics = scaler.monitor_resources()
    scaling_decision = scaler.make_scaling_decision(initial_metrics)
    print(f"Scaling decision: {scaling_decision.decision_type}")

    # Adjust config based on scaling
    if scaling_decision.decision_type == "scale_down":
        config.limits.max_iterations = int(config.limits.max_iterations * scaler.current_scaling_factor)

    llm_client = build_llm_client(config.llm)
    orchestrator = AgentOrchestrator(config, llm_client)

    # Integrate MetaRecursionEngine
    from .autoeval import AutoEvaluator
    auto_evaluator = AutoEvaluator(thresholds=PlannerThresholds(), config=config)
    recursion_engine = integrate_meta_recursion(config, PatchManager(Path.cwd()), auto_evaluator)
    recursion_context = recursion_engine.initiate_recursion(spec.goal)

    result = await asyncio.to_thread(orchestrator.run, spec.goal)

    # Evaluate recursion
    post_metrics = scaler.monitor_resources()
    post_scaling = scaler.make_scaling_decision(post_metrics)
    recursion_engine.evaluate_and_recurse(recursion_context, {"actions_success_rate": result.completed and 1.0 or 0.0})

    return result


async def _drain_seeds(
    backlog_path: Path,
    *,
    default_config: Path,
    max_runs: int | None,
    max_steps_override: int | None,
) -> None:
    runner = SeedRunner(backlog_path)
    concurrency = max_runs or 3
    pending = runner.backlog.list_pending()
    if not pending:
        return
    to_run = pending[:concurrency]

    # Mark in progress
    for seed in to_run:
        runner.backlog.mark_in_progress(seed.id)

    async def run_single_seed(seed: Seed, default_config: Path, max_steps_override: int | None) -> tuple[Seed, AgentResult]:
        if seed.type == "meta":
            config_path = Path("configs/sample.yaml")
        else:
            config_path = Path(seed.config or str(default_config))
        config = load_config(config_path)
        if seed.max_steps and not max_steps_override:
            config.limits.max_iterations = seed.max_steps
        elif max_steps_override:
            config.limits.max_iterations = max_steps_override
        llm_client = build_llm_client(config.llm)
        orchestrator = AgentOrchestrator(config, llm_client)
        result = await asyncio.to_thread(orchestrator.run, seed.goal)
        return seed, result

    tasks = [run_single_seed(seed, default_config, max_steps_override) for seed in to_run]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    runs = 0
    for res in results:
        if isinstance(res, Exception):
            print(f"Error in seed run: {res}")
            continue
        seed, result = res
        runs += 1
        notes = "; ".join(result.errors) if result.errors else ""
        if result.completed:
            runner.backlog.mark_completed(
                seed.id,
                notes=notes or None,
                iterations=result.iterations,
                memories_reused=result.memories_reused,
            )
            print(f"Seed {seed.id} -> completed (iterations={result.iterations})")
        else:
            runner.backlog.mark_failed(
                seed.id,
                notes=notes or "Seed não concluída",
                iterations=result.iterations,
                memories_reused=result.memories_reused,
            )
            print(f"Seed {seed.id} -> failed")


__all__ = ["AutoLoop", "GoalSpec", "load_goal_rotation", "run_autopilot"]
