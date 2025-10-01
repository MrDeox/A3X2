"""Integration-style tests for the autopilot loop helpers."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from a3x.autoloop import GoalSpec, run_autopilot


def _build_event(type_name: str, *, success: bool, command: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        action=SimpleNamespace(
            type=SimpleNamespace(name=type_name),
            command=command,
        ),
        observation=SimpleNamespace(success=success),
    )


def test_run_autopilot_single_cycle(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy", encoding="utf-8")
    backlog_path = tmp_path / "backlog.yaml"

    run_result = SimpleNamespace(
        completed=True,
        iterations=2,
        failures=0,
        history=SimpleNamespace(
            events=[
                _build_event("RUN_COMMAND", success=True, command=["pytest", "-q"]),
                _build_event("APPLY_PATCH", success=True),
            ]
        ),
    )

    mock_backlog = Mock()
    mock_auto_seeder = Mock()
    mock_auto_seeder.monitor_and_seed.return_value = []

    with patch("a3x.autoloop.SeedBacklog.load", return_value=mock_backlog), patch(
        "a3x.autoloop.AutoSeeder", return_value=mock_auto_seeder
    ), patch("a3x.autoloop._run_goal", return_value=run_result) as mock_run_goal, patch(
        "a3x.autoloop._drain_seeds"
    ) as mock_drain:
        goals = [GoalSpec(goal="Test goal", config=config_path)]
        exit_code = run_autopilot(
            goals,
            cycles=1,
            backlog_path=backlog_path,
            seed_default_config=config_path,
        )

    assert exit_code == 0
    mock_run_goal.assert_called_once_with(goals[0])
    mock_auto_seeder.monitor_and_seed.assert_called_once()
    mock_drain.assert_called_once()

    metrics_passed = mock_auto_seeder.monitor_and_seed.call_args.args[0]
    assert metrics_passed["actions_success_rate"] == pytest.approx(1.0)
    assert metrics_passed["apply_patch_success_rate"] == pytest.approx(1.0)
    assert metrics_passed["tests_success_rate"] == pytest.approx(1.0)
