"""Integration coverage for multiple autopilot cycles."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from a3x.autoloop import GoalSpec, run_autopilot
from a3x.seeds import Seed


def _event(type_name: str, *, success: bool) -> SimpleNamespace:
    return SimpleNamespace(
        action=SimpleNamespace(
            type=SimpleNamespace(name=type_name),
            command=["pytest"] if type_name == "RUN_COMMAND" else None,
        ),
        observation=SimpleNamespace(success=success),
    )


def test_run_autopilot_multiple_cycles(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy", encoding="utf-8")
    backlog_path = tmp_path / "backlog.yaml"

    success_run = SimpleNamespace(
        completed=True,
        iterations=1,
        failures=0,
        history=SimpleNamespace(events=[_event("RUN_COMMAND", success=True)]),
    )
    failed_run = SimpleNamespace(
        completed=False,
        iterations=1,
        failures=1,
        history=SimpleNamespace(events=[_event("APPLY_PATCH", success=False)]),
    )

    mock_backlog = Mock()
    mock_backlog.exists.return_value = False
    mock_auto_seeder = Mock()
    seed_generated = Seed(id="seed-123", goal="Melhorar estabilidade de testes")
    mock_auto_seeder.monitor_and_seed.side_effect = [[], [seed_generated]]

    with patch("a3x.autoloop.SeedBacklog.load", return_value=mock_backlog), patch(
        "a3x.autoloop.AutoSeeder", return_value=mock_auto_seeder
    ), patch("a3x.autoloop._run_goal", side_effect=[success_run, failed_run]) as mock_run_goal, patch(
        "a3x.autoloop._drain_seeds"
    ) as mock_drain:
        goals = [GoalSpec(goal="Goal A", config=config_path)]
        exit_code = run_autopilot(
            goals,
            cycles=2,
            backlog_path=backlog_path,
            seed_default_config=config_path,
        )

    assert exit_code == 1
    assert mock_run_goal.call_count == 2
    assert mock_auto_seeder.monitor_and_seed.call_count == 2
    mock_backlog.add_seed.assert_called_once_with(seed_generated)
    assert mock_drain.call_count == 2

    metrics_second = mock_auto_seeder.monitor_and_seed.call_args_list[1].args[0]
    assert metrics_second["actions_success_rate"] == pytest.approx(0.0)
    assert metrics_second["apply_patch_success_rate"] == pytest.approx(0.0)
