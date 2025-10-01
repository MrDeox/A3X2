"""Auto-generated E2E test for multi-cycle autoloop with seeding.
AUTO-GENERATED. Edit via E2ETestGenerator."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from a3x.autoloop import AutoLoop
from a3x.llm import LLMClient
from a3x.executor import ActionExecutor
from a3x.seeds import SeedBacklog
from a3x.autoeval import AutoEvaluator


@pytest.fixture
def mock_llm_client():
    client = Mock(spec=LLMClient)
    # Simulate varying success: first cycles succeed, last triggers seed on low threshold
    def side_effect(*args, **kwargs):
        if kwargs.get("iteration", 0) < 2:
            return {"action": "write_file", "params": {"path": "test.py", "content": "print('test')"}}
        else:
            return {"action": "noop", "params": {}}  # Low success trigger
    client.generate_action.side_effect = side_effect
    return client


@pytest.fixture
def mock_executor():
    executor = Mock(spec=ActionExecutor)
    executor.execute.side_effect = [Observation(success=True)] * 2 + [Observation(success=False)]
    return executor


@pytest.fixture
def mock_backlog(tmp_path: Path):
    backlog_path = tmp_path / "backlog.yaml"
    backlog = Mock(spec=SeedBacklog)
    backlog.path = backlog_path
    backlog.add_seed.return_value = None
    return backlog


@pytest.fixture
def mock_evaluator(tmp_path: Path):
    evaluator = Mock(spec=AutoEvaluator)
    evaluator.record.return_value = Mock(completed=False, metrics={"actions_success_rate": 0.7})
    return evaluator


def test_multi_cycle_with_seeding(mock_llm_client, mock_executor, mock_backlog, mock_evaluator, tmp_path: Path):
    with patch("a3x.autoloop.LLMClient", return_value=mock_llm_client),          patch("a3x.autoloop.Executor", return_value=mock_executor),          patch("a3x.autoloop.SeedBacklog", return_value=mock_backlog),          patch("a3x.autoloop.AutoEvaluator", return_value=mock_evaluator),          patch("a3x.autoloop.config.BASE_DIR", tmp_path):
        loop = AutoLoop(goal="test multi-cycle", max_iterations=3)
        result = loop.run()
    
    assert result.iterations == 3
    assert result.metrics.get("actions_success_rate", 0) < 0.8  # Triggers seeding
    mock_backlog.add_seed.assert_called()  # Seeding on low threshold
    mock_evaluator.record.assert_called()
