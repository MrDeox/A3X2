"""Integration facade tests for the SeedRunner orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

from a3x.seed_runner import SeedRunner, SeedRunResult
from a3x.seeds import Seed


def _build_seed(**overrides) -> Seed:
    defaults = {
        "id": "seed-1",
        "goal": "Implementar endpoint /health",
        "priority": "high",
        "status": "pending",
        "type": "generic",
    }
    defaults.update(overrides)
    return Seed(**defaults)


def test_seed_runner_success_flow(tmp_path: Path) -> None:
    backlog_mock = Mock()
    backlog_mock.next_seed.return_value = _build_seed()

    runner = SeedRunner.__new__(SeedRunner)
    runner.backlog = backlog_mock

    orchestrator_result = SimpleNamespace(
        completed=True,
        errors=[],
        iterations=4,
        memories_reused=2,
    )

    with patch("a3x.seed_runner.load_config") as mock_load_config, patch(
        "a3x.seed_runner.build_llm_client"
    ) as mock_build_llm, patch("a3x.seed_runner.AgentOrchestrator") as mock_orchestrator:
        mock_load_config.return_value = SimpleNamespace(
            limits=SimpleNamespace(max_iterations=5),
            llm=Mock(),
        )
        mock_build_llm.return_value = Mock()
        mock_instance = mock_orchestrator.return_value
        mock_instance.run.return_value = orchestrator_result

        result = SeedRunner.run_next(
            runner,
            default_config=tmp_path / "config.yaml",
        )

    assert isinstance(result, SeedRunResult)
    assert result.completed is True
    assert result.iterations == orchestrator_result.iterations
    assert result.memories_reused == orchestrator_result.memories_reused
    backlog_mock.mark_in_progress.assert_called_once()
    backlog_mock.mark_completed.assert_called_once_with(
        "seed-1",
        notes="",
        iterations=orchestrator_result.iterations,
        memories_reused=orchestrator_result.memories_reused,
    )


def test_seed_runner_failure_marks_seed(tmp_path: Path) -> None:
    failing_seed = _build_seed(id="seed-2")
    backlog_mock = Mock()
    backlog_mock.next_seed.return_value = failing_seed

    runner = SeedRunner.__new__(SeedRunner)
    runner.backlog = backlog_mock

    orchestrator_result = SimpleNamespace(
        completed=False,
        errors=["timeout"],
        iterations=7,
        memories_reused=0,
    )

    with patch("a3x.seed_runner.load_config") as mock_load_config, patch(
        "a3x.seed_runner.build_llm_client"
    ) as mock_build_llm, patch("a3x.seed_runner.AgentOrchestrator") as mock_orchestrator:
        mock_load_config.return_value = SimpleNamespace(
            limits=SimpleNamespace(max_iterations=5),
            llm=Mock(),
        )
        mock_build_llm.return_value = Mock()
        mock_orchestrator.return_value.run.return_value = orchestrator_result

        result = SeedRunner.run_next(
            runner,
            default_config=tmp_path / "config.yaml",
        )

    assert isinstance(result, SeedRunResult)
    assert result.completed is False
    assert result.notes == "timeout"
    assert result.iterations == orchestrator_result.iterations
    backlog_mock.mark_failed.assert_called_once_with(
        "seed-2",
        notes="timeout",
        iterations=orchestrator_result.iterations,
        memories_reused=orchestrator_result.memories_reused,
    )
    backlog_mock.mark_completed.assert_not_called()
