"""Auto-generated E2E test for basic autoloop cycle.
AUTO-GENERATED. Edit via E2ETestGenerator."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from a3x.autoloop import AutoLoop
from a3x.llm import LLMClient
from a3x.executor import ActionExecutor
from a3x.seeds import SeedBacklog


@pytest.fixture
def mock_llm_client():
    client = Mock(spec=LLMClient)
    client.generate_action.return_value = {"action": "write_file", "params": {"path": "test.py", "content": "print('test')"}}
    return client


@pytest.fixture
def mock_executor():
    executor = Mock(spec=ActionExecutor)
    executor.execute.return_value = Observation(success=True, output="Mock execution")
    return executor


@pytest.fixture
def mock_backlog():
    backlog = Mock(spec=SeedBacklog)
    backlog.load.return_value = {"goals": ["test goal"]}
    return backlog


def test_basic_autoloop_cycle(mock_llm_client, mock_executor, mock_backlog, tmp_path: Path):
    with patch("a3x.autoloop.LLMClient", return_value=mock_llm_client),          patch("a3x.autoloop.Executor", return_value=mock_executor),          patch("a3x.autoloop.SeedBacklog", return_value=mock_backlog),          patch("a3x.autoloop.config.BASE_DIR", tmp_path):
        loop = AutoLoop(goal="test goal", max_iterations=1)
        result = loop.run()
    
    assert result.completed
    assert result.metrics.get("actions_success_rate", 0) > 0.8
    mock_executor.execute.assert_called()
    mock_llm_client.generate_action.assert_called()
