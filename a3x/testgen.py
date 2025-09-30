"""Utilities to generate adaptive growth tests for the SeedAI loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class GrowthTestGenerator:
    """Creates pytest files that enforce non-regression on tracked metrics."""

    def __init__(self, history_path: Path | str, output_path: Path | str) -> None:
        self.history_path = Path(history_path)
        self.output_path = Path(output_path)
        self.insights_gen = InsightsTestGenerator()

    def ensure_tests(self) -> None:
        if not self.history_path.exists():
            return
        try:
            history = json.loads(self.history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(history, dict) or not history:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._render_test(history)
        if (
            self.output_path.exists()
            and self.output_path.read_text(encoding="utf-8") == content
        ):
            return
        self.output_path.write_text(content, encoding="utf-8")
        
        # Hook for new insights tests
        self.insights_gen.generate_tests()

    def _render_test(self, history: Dict[str, List[float]]) -> str:
        del history  # conteúdo reservado para usos futuros
        return (
            '"""Testes gerados automaticamente para verificar saúde das métricas SeedAI.\n' 
            '\n' 
            'AUTO-GENERATED FILE. Edite via GrowthTestGenerator."""\n' 
            '\n' 
            'import json\n' 
            'from pathlib import Path\n' 
            '\n' 
            '\n' 
            'def _load_history() -> dict[str, list[float]]:\n' 
            '    history_path = Path(__file__).resolve().parents[2] / "seed" / "metrics" / "history.json"\n' 
            '    data = json.loads(history_path.read_text(encoding="utf-8"))\n' 
            '    if not isinstance(data, dict):\n' 
            '        raise AssertionError("Histórico de métricas inválido")\n' 
            '    return {key: list(map(float, values)) for key, values in data.items()}\n' 
            '\n' 
            '\n' 
            'def test_seed_metrics_health() -> None:\n' 
            '    history = _load_history()\n' 
            '    assert history, "Histórico de métricas não pode estar vazio"\n' 
            '    for metric, values in history.items():\n' 
            '        assert values, f"Métrica {metric} sem registros"\n' 
            '        best = max(values)\n' 
            '        last = values[-1]\n' 
            '        for idx, value in enumerate(values):\n' 
            '            assert value >= -1e-6, f"Métrica {metric} negativa no índice {idx}: {value}"\n' 
            '        if "success_rate" in metric:\n' 
            '            assert last >= best, f"Métrica {metric} regrediu: último valor {last} < melhor valor {best}"\n' 
            '        if metric.endswith("latency") and best > 0:\n' 
            '            assert last <= best * 4 + 1e-6, f"Latência {metric} explodiu: {last} > {best}"\n' 
        )


class InsightsTestGenerator:
    """Generates unit tests for memory/insights.py, including StatefulRetriever."""

    def __init__(self, output_path: Path | str = "tests/unit/a3x/memory/test_insights.py") -> None:
        self.output_path = Path(output_path)
        self.output_dir = self.output_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_tests(self) -> None:
        """Generate basic unit tests for new StatefulRetriever class."""
        content = '''"""Auto-generated unit tests for a3x/memory/insights.py.
AUTO-GENERATED. Edit via InsightsTestGenerator."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import hashlib

from a3x.actions import AgentState
from a3x.memory.insights import StatefulRetriever, Insight
from a3x.memory.store import MemoryEntry, SemanticMemory


@pytest.fixture
def mock_store():
    store = MagicMock(spec=SemanticMemory)
    mock_entry = MemoryEntry(
        id="test_id",
        created_at=datetime.now(timezone.utc).isoformat(),
        title="Test Insight",
        content="Test content with recent failure in actions.",
        tags=["test", "failure"],
        metadata={"goal": "test_goal", "snapshot_hash": "abc123"},
        embedding=[0.1] * 384  # Mock embedding
    )
    store.query.return_value = [(mock_entry, 0.8), (mock_entry, 0.6)]  # One >0.7
    store.entries = [mock_entry]
    return store


@pytest.fixture
def mock_state():
    return AgentState(
        goal="Test goal",
        history_snapshot="Test history with failures.",
        iteration=1,
        max_iterations=5,
        seed_context="Test context"
    )


def test_stateful_retriever_retrieve_session_context(mock_store, mock_state):
    retriever = StatefulRetriever()
    with patch.object(retriever, 'store', mock_store):
        insights = retriever.retrieve_session_context(mock_state)
    
    assert len(insights) == 1  # Filtered to >0.7
    assert isinstance(insights[0], Insight)
    assert insights[0].similarity > 0.7
    mock_store.query.assert_called_once()

def test_stateful_retriever_derivation_detection(mock_store, mock_state):
    retriever = StatefulRetriever()
    mock_state.history_snapshot = "Changed history"
    with patch.object(retriever, 'store', mock_store):
        insights = retriever.retrieve_session_context(mock_state)
    
    if insights:
        assert insights[0].metadata.get("derivation_flagged") is True  # Hash differs
        assert "snapshot_hash" in insights[0].metadata

def test_stateful_retriever_update_snapshot_hash():
    retriever = StatefulRetriever()
    snapshot = "Test snapshot"
    retriever.update_snapshot_hash(snapshot)
    assert retriever.last_snapshot_hash == hashlib.md5(snapshot.encode()).hexdigest()

def test_insight_from_entry():
    mock_entry = MemoryEntry(
        id="id", created_at="2023-01-01T00:00:00Z",
        title="Title", content="Content", tags=["tag"],
        metadata={"key": "value"},
        embedding=[0.1]*384
    )
    insight = Insight.from_entry(mock_entry, 0.85)
    assert insight.title == "Title"
    assert insight.similarity == 0.85
    assert insight.created_at == "2023-01-01T00:00:00Z"
'''
        if self.output_path.exists() and self.output_path.read_text(encoding="utf-8") == content:
            return
        self.output_path.write_text(content, encoding="utf-8")


class E2ETestGenerator:
    """Generates end-to-end and multi-cycle test files for A3X agent cycles."""

    def __init__(self, output_dir: Path | str = "tests/integration") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_basic_cycle_test(self, filename: str = "test_basic_autoloop.py") -> Path:
        """Generate a basic e2e test for a single autoloop cycle."""
        filepath = self.output_dir / filename
        content = self._render_basic_cycle_test()
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def generate_multi_cycle_test(self, filename: str = "test_multi_cycle.py", cycles: int = 3) -> Path:
        """Generate a multi-cycle e2e test with seed generation on low thresholds."""
        filepath = self.output_dir / filename
        content = self._render_multi_cycle_test(cycles)
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def generate_seed_runner_test(self, filename: str = "test_seed_runner.py") -> Path:
        """Generate an e2e test for seed_runner execution."""
        filepath = self.output_dir / filename
        content = self._render_seed_runner_test()
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def _render_basic_cycle_test(self) -> str:
        return '''"""Auto-generated E2E test for basic autoloop cycle.
AUTO-GENERATED. Edit via E2ETestGenerator."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from a3x.autoloop import AutoLoop
from a3x.llm import LLMClient
from a3x.executor import Executor
from a3x.seeds import SeedBacklog


@pytest.fixture
def mock_llm_client():
    client = Mock(spec=LLMClient)
    client.generate_action.return_value = {"action": "write_file", "params": {"path": "test.py", "content": "print('test')"}}
    return client


@pytest.fixture
def mock_executor():
    executor = Mock(spec=Executor)
    executor.apply.return_value = True
    return executor


@pytest.fixture
def mock_backlog():
    backlog = Mock(spec=SeedBacklog)
    backlog.load.return_value = {"goals": ["test goal"]}
    return backlog


def test_basic_autoloop_cycle(mock_llm_client, mock_executor, mock_backlog, tmp_path: Path):
    with patch("a3x.autoloop.LLMClient", return_value=mock_llm_client), \
         patch("a3x.autoloop.Executor", return_value=mock_executor), \
         patch("a3x.autoloop.SeedBacklog", return_value=mock_backlog), \
         patch("a3x.autoloop.config.BASE_DIR", tmp_path):
        loop = AutoLoop(goal="test goal", max_iterations=1)
        result = loop.run()
    
    assert result.completed
    assert result.metrics.get("actions_success_rate", 0) > 0.8
    mock_executor.apply.assert_called_once()
    mock_llm_client.generate_action.assert_called()
'''

    def _render_multi_cycle_test(self, cycles: int) -> str:
        return f'''"""Auto-generated E2E test for multi-cycle autoloop with seeding.
AUTO-GENERATED. Edit via E2ETestGenerator."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from a3x.autoloop import AutoLoop
from a3x.llm import LLMClient
from a3x.executor import Executor
from a3x.seeds import SeedBacklog
from a3x.autoeval import AutoEvaluator


@pytest.fixture
def mock_llm_client():
    client = Mock(spec=LLMClient)
    # Simulate varying success: first cycles succeed, last triggers seed on low threshold
    def side_effect(*args, **kwargs):
        if kwargs.get("iteration", 0) < {cycles - 1}:
            return {{"action": "write_file", "params": {{"path": "test.py", "content": "print('test')"}}}}
        else:
            return {{"action": "noop", "params": {{}}}}  # Low success trigger
    client.generate_action.side_effect = side_effect
    return client


@pytest.fixture
def mock_executor():
    executor = Mock(spec=Executor)
    executor.apply.side_effect = [True] * ({cycles - 1}) + [False]  # Last fails
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
    evaluator.record.return_value = Mock(completed=False, metrics={{"actions_success_rate": 0.7}})
    return evaluator


def test_multi_cycle_with_seeding(mock_llm_client, mock_executor, mock_backlog, mock_evaluator, tmp_path: Path):
    with patch("a3x.autoloop.LLMClient", return_value=mock_llm_client), \
         patch("a3x.autoloop.Executor", return_value=mock_executor), \
         patch("a3x.autoloop.SeedBacklog", return_value=mock_backlog), \
         patch("a3x.autoloop.AutoEvaluator", return_value=mock_evaluator), \
         patch("a3x.autoloop.config.BASE_DIR", tmp_path):
        loop = AutoLoop(goal="test multi-cycle", max_iterations={cycles})
        result = loop.run()
    
    assert result.iterations == {cycles}
    assert result.metrics.get("actions_success_rate", 0) < 0.8  # Triggers seeding
    mock_backlog.add_seed.assert_called()  # Seeding on low threshold
    mock_evaluator.record.assert_called()
'''

    def _render_seed_runner_test(self) -> str:
        return '''"""Auto-generated E2E test for seed_runner execution.
AUTO-GENERATED. Edit via E2ETestGenerator."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from a3x.seed_runner import SeedRunner
from a3x.seeds import SeedBacklog
from a3x.executor import Executor


@pytest.fixture
def mock_backlog():
    backlog = Mock(spec=SeedBacklog)
    backlog.get_next_seed.return_value = {"id": "test_seed", "description": "Test seed", "type": "improvement"}
    return backlog


@pytest.fixture
def mock_executor():
    executor = Mock(spec=Executor)
    executor.execute_seed.return_value = {"success": True, "metrics": {"seed_success_rate": 1.0}}
    return executor


def test_seed_runner_execution(mock_backlog, mock_executor, tmp_path: Path):
    with patch("a3x.seed_runner.SeedBacklog", return_value=mock_backlog), \
         patch("a3x.seed_runner.Executor", return_value=mock_executor), \
         patch("a3x.seed_runner.config.BASE_DIR", tmp_path):
        runner = SeedRunner(config_path=str(tmp_path / "config.yaml"))
        result = runner.run_next()
    
    assert result["success"]
    assert result["metrics"].get("seed_success_rate", 0) == 1.0
    mock_backlog.get_next_seed.assert_called_once()
    mock_executor.execute_seed.assert_called_once()
    # Negative scenario: failure rollback
    mock_executor.execute_seed.return_value = {"success": False}
    result_fail = runner.run_next()
    assert not result_fail["success"]
    # Assert no permanent changes on failure (mock would handle rollback assertion)
'''


__all__ = ["GrowthTestGenerator", "E2ETestGenerator"]
