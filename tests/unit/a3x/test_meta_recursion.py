"""Tests for the MetaRecursionEngine in A3X."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from a3x.autoeval import AutoEvaluator
from a3x.config import AgentConfig
from a3x.meta_recursion import MetaRecursionEngine, RecursionContext
from a3x.patch import PatchManager


class TestMetaRecursionEngine:
    """Tests for MetaRecursionEngine."""

    def setup_method(self):
        """Setup test fixtures."""
        self.workspace_path = Path("tests/temp_workspace")
        self.workspace_path.mkdir(exist_ok=True)

        # Mock config
        self.config = AgentConfig(
            llm=Mock(),
            workspace=Mock(root=str(self.workspace_path)),
            limits=Mock(),
            tests=Mock(),
            policies=Mock(),
            goals=Mock(),
            loop=Mock(),
            audit=Mock()
        )
        self.config.get.return_value = 10  # max_depth=10

        # Mock dependencies
        self.patch_manager = Mock(spec=PatchManager)
        self.auto_evaluator = Mock(spec=AutoEvaluator)

        self.engine = MetaRecursionEngine(
            self.config, self.patch_manager, self.auto_evaluator
        )
        self.engine.max_depth = 10

    def teardown_method(self):
        """Teardown test fixtures."""
        import shutil
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)

    def test_engine_initialization(self):
        """Test engine initialization with max_depth=10."""
        assert self.engine.max_depth == 10
        assert self.engine.current_depth == 0
        assert isinstance(self.engine.context_stack, list)
        assert len(self.engine.context_stack) == 0

    def test_initiate_recursion_within_depth(self):
        """Test initiating recursion within max depth."""
        goal = "Test goal"
        context = self.engine.initiate_recursion(goal)

        assert isinstance(context, RecursionContext)
        assert context.depth == 1
        assert context.goal == goal
        assert context.status == "active"
        assert len(self.engine.context_stack) == 1
        assert self.engine.current_depth == 1

        # Verify file saved
        context_file = self.engine.recursion_path / f"{context.id}.json"
        assert context_file.exists()
        with open(context_file) as f:
            saved = json.load(f)
        assert saved["id"] == context.id
        assert saved["depth"] == 1

    def test_initiate_recursion_exceeds_max_depth(self):
        """Test initiating recursion exceeding max depth raises ValueError."""
        # Set current depth to max
        self.engine.current_depth = 10

        goal = "Exceed depth goal"
        with pytest.raises(ValueError, match="Maximum recursion depth 10 reached"):
            self.engine.initiate_recursion(goal)

    @patch("time.sleep")
    def test_backoff_on_depth_exceed_in_agent_init(self, mock_sleep):
        """Test exponential backoff when depth exceeds 10 in agent init (integrated test)."""
        # This tests the integration point in agent.py __init__
        # Mock AgentOrchestrator init with depth=11
        with patch("a3x.agent.AgentOrchestrator.__init__"):
            from a3x.agent import AgentOrchestrator
            agent = AgentOrchestrator(self.config, Mock(), depth=11)

        # Verify backoff called with appropriate delay
        backoff = min(60, 2 ** (11 - 10))  # 2^1 = 2s
        mock_sleep.assert_called_once_with(backoff)

    @patch("time.sleep")
    def test_backoff_capped_at_60s(self, mock_sleep):
        """Test backoff capped at 60s for very deep recursion."""
        # Simulate depth=20
        with patch("a3x.agent.AgentOrchestrator.__init__"):
            from a3x.agent import AgentOrchestrator
            agent = AgentOrchestrator(self.config, Mock(), depth=20)

        # 2^(20-10) = 1024, but capped at 60
        mock_sleep.assert_called_once_with(60)

    def test_evaluate_and_recurse_continue(self):
        """Test evaluation continues recursion on sufficient improvement."""
        context = self.engine.initiate_recursion("Initial goal")
        metrics = {"actions_success_rate": 0.2}  # Above threshold 0.1

        with patch.object(self.engine, "initiate_recursion") as mock_initiate:
            recurse = self.engine.evaluate_and_recurse(context, metrics)

        assert recurse is True
        mock_initiate.assert_called_once()
        assert context.status == "active"  # Not completed yet

    def test_evaluate_and_recurse_stop(self):
        """Test evaluation stops recursion on insufficient improvement."""
        context = self.engine.initiate_recursion("Initial goal")
        metrics = {"actions_success_rate": 0.05}  # Below threshold 0.1

        recurse = self.engine.evaluate_and_recurse(context, metrics)

        assert recurse is False
        assert context.status == "completed"
        assert self.engine.current_depth == 0
        assert len(self.engine.context_stack) == 0

    def test_apply_recursive_patch_success(self):
        """Test applying recursive patch successfully."""
        context = self.engine.initiate_recursion("Patch goal")
        diff = "sample diff"
        self.patch_manager.apply.return_value = (True, "Patch success")

        success = self.engine.apply_recursive_patch(diff, context)

        assert success is True
        self.patch_manager.apply.assert_called_once_with(diff)
        self.auto_evaluator.record_metric.assert_called_with("recursion.patch_success", 1.0)
        assert "Patch applied" in context.improvements_applied[0]

    def test_apply_recursive_patch_failure(self):
        """Test applying recursive patch failure."""
        context = self.engine.initiate_recursion("Patch goal")
        diff = "invalid diff"
        self.patch_manager.apply.return_value = (False, "Patch failed")

        success = self.engine.apply_recursive_patch(diff, context)

        assert success is False
        self.auto_evaluator.record_metric.assert_called_with("recursion.patch_success", 0.0)
        assert context.status == "failed"

    def test_complete_context(self):
        """Test completing a recursion context."""
        context = self.engine.initiate_recursion("Test goal")
        self.engine._complete_context(context)

        assert context.status == "completed"
        assert len(self.engine.recursion_history) == 1
        history_file = self.engine.recursion_path / "history.json"
        assert history_file.exists()

    def test_get_recursion_summary(self):
        """Test getting recursion summary."""
        # Initiate one context
        self.engine.initiate_recursion("Summary goal")
        self.engine._complete_context(self.engine.context_stack[0])

        summary = self.engine.get_recursion_summary()

        assert summary["current_depth"] == 0
        assert summary["active_contexts"] == 0
        assert summary["completed_contexts"] == 1
        assert summary["max_depth_reached"] == 1
        assert isinstance(summary["avg_improvement"], float)
