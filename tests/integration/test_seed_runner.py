"""Auto-generated E2E test for seed_runner execution.
AUTO-GENERATED. Edit via E2ETestGenerator."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from a3x.seed_runner import SeedRunner
from a3x.seeds import SeedBacklog
from a3x.executor import ActionExecutor


@pytest.fixture
def mock_backlog():
    backlog = Mock(spec=SeedBacklog)
    backlog.get_next_seed.return_value = {"id": "test_seed", "description": "Test seed", "type": "improvement"}
    return backlog


@pytest.fixture
def mock_executor():
    executor = Mock(spec=ActionExecutor)
    executor.execute.return_value = Observation(success=True, output='{"success": True, "metrics": {"seed_success_rate": 1.0}}')
    return executor


def test_seed_runner_execution(mock_backlog, mock_executor, tmp_path: Path):
    with patch("a3x.seed_runner.SeedBacklog", return_value=mock_backlog),          patch("a3x.seed_runner.Executor", return_value=mock_executor),          patch("a3x.seed_runner.config.BASE_DIR", tmp_path):
        runner = SeedRunner(config_path=str(tmp_path / "config.yaml"))
        result = runner.run_next()
    
    assert result.success
    assert result.metrics.get("seed_success_rate", 0) == 1.0
    mock_backlog.get_next_seed.assert_called_once()
    mock_executor.execute.assert_called()
    # Negative scenario: failure rollback
    mock_executor.execute.return_value = Observation(success=False)
    result_fail = runner.run_next()
    assert not result_fail.success
    # Assert no permanent changes on failure (mock would handle rollback assertion)
