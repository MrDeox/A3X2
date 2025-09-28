"""Autonomous loop that alternates between high-level goals and seeds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from .agent import AgentOrchestrator
from .config import load_config
from .llm import build_llm_client
from .seed_runner import SeedRunner


@dataclass
class GoalSpec:
    goal: str
    config: Path
    max_steps: Optional[int] = None


def load_goal_rotation(path: str | Path) -> List[GoalSpec]:
    rotation_path = Path(path)
    if not rotation_path.exists():
        raise FileNotFoundError(f"Goal rotation file not found: {rotation_path}")
    raw = yaml.safe_load(rotation_path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError("Goal rotation file must contain a list")

    specs: List[GoalSpec] = []
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


def run_autopilot(
    goals: List[GoalSpec],
    *,
    cycles: int,
    backlog_path: str | Path,
    seed_default_config: str | Path,
    seed_max: Optional[int] = None,
    seed_max_steps: Optional[int] = None,
) -> int:
    if not goals:
        raise ValueError("Goal rotation list cannot be empty")

    backlog_path = Path(backlog_path)
    seed_default_config = Path(seed_default_config)
    backlog_path.parent.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for cycle in range(cycles):
        spec = goals[cycle % len(goals)]
        print(f"=== Cycle {cycle + 1} :: {spec.goal} ===")
        run_result = _run_goal(spec)
        print(
            f"Resultado: goal='{spec.goal}', completed={run_result.completed}, iterations={run_result.iterations}, failures={run_result.failures}"
        )
        exit_code = 0 if run_result.completed else 1
        _drain_seeds(
            backlog_path,
            default_config=seed_default_config,
            max_runs=seed_max,
            max_steps_override=seed_max_steps,
        )
    return exit_code


def _run_goal(spec: GoalSpec):
    config = load_config(spec.config)
    if spec.max_steps is not None:
        config.limits.max_iterations = spec.max_steps
    llm_client = build_llm_client(config.llm)
    orchestrator = AgentOrchestrator(config, llm_client)
    return orchestrator.run(spec.goal)


def _drain_seeds(
    backlog_path: Path,
    *,
    default_config: Path,
    max_runs: Optional[int],
    max_steps_override: Optional[int],
) -> None:
    runner = SeedRunner(backlog_path)
    runs = 0
    while True:
        if max_runs is not None and runs >= max_runs:
            break
        result = runner.run_next(
            default_config=default_config, max_steps_override=max_steps_override
        )
        if result is None:
            break
        runs += 1
        print(
            f"Seed {result.seed_id} -> {result.status} (completed={result.completed})"
        )


__all__ = ["GoalSpec", "load_goal_rotation", "run_autopilot"]
