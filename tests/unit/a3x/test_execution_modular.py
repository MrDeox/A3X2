"""Comprehensive tests for the new modular execution system.

This module tests the refactored execution components in isolation
and integration to ensure the modular architecture works correctly.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from a3x.config import AgentConfig, LLMConfig, WorkspaceConfig, LimitsConfig, TestSettings, PoliciesConfig, GoalsConfig, LoopConfig, AuditConfig
from a3x.actions import AgentAction, ActionType, Observation
from a3x.execution import ExecutionOrchestrator
from a3x.execution.actions import ActionHandlers
from a3x.execution.validation import ValidationEngine
from a3x.execution.rollback import RollbackManager
from a3x.execution.analysis import CodeAnalyzer
from a3x.execution.safety import SafetyMonitor
from a3x.execution.monitoring import PerformanceMonitor


@pytest.fixture
def test_config():
    """Create a test configuration for the execution system."""
    return AgentConfig(
        llm=LLMConfig(type='manual'),
        workspace=WorkspaceConfig(root=Path('.')),
        limits=LimitsConfig(),
        tests=TestSettings(),
        policies=PoliciesConfig(),
        goals=GoalsConfig(),
        loop=LoopConfig(),
        audit=AuditConfig()
    )


@pytest.fixture
def orchestrator(test_config):
    """Create an ExecutionOrchestrator instance for testing."""
    return ExecutionOrchestrator(test_config)


class TestExecutionOrchestrator:
    """Test the main execution orchestrator."""

    def test_orchestrator_initialization(self, test_config):
        """Test that the orchestrator initializes correctly."""
        orchestrator = ExecutionOrchestrator(test_config)

        assert orchestrator.config == test_config
        assert orchestrator.workspace_root == Path('.').resolve()
        assert orchestrator.patch_manager is not None
        assert orchestrator.change_logger is not None

    def test_component_status(self, orchestrator):
        """Test that component status is retrieved correctly."""
        status = orchestrator.get_component_status()

        assert "action_handlers" in status
        assert "validation_engine" in status
        assert "rollback_manager" in status
        assert "code_analyzer" in status
        assert "safety_monitor" in status
        assert "performance_monitor" in status

    def test_message_action_execution(self, orchestrator):
        """Test execution of a simple message action."""
        action = AgentAction(type=ActionType.MESSAGE, text="Test message")
        result = orchestrator.execute(action)

        assert result.success is True
        assert result.output == "Test message"
        assert result.type == "message"

    def test_workspace_path_resolution(self, test_config):
        """Test workspace path resolution."""
        orchestrator = ExecutionOrchestrator(test_config)

        # Test relative path
        rel_path = orchestrator.get_workspace_path("test.txt")
        assert isinstance(rel_path, Path)
        assert not rel_path.is_absolute()

        # Test absolute path (should be allowed in test config)
        abs_path = orchestrator.get_workspace_path("/tmp/test.txt")
        assert abs_path == Path("/tmp/test.txt").resolve()


class TestActionHandlers:
    """Test the action handlers component."""

    def test_action_handlers_initialization(self, orchestrator):
        """Test that action handlers initialize correctly."""
        handlers = ActionHandlers(orchestrator)

        assert handlers.orchestrator == orchestrator
        assert handlers.workspace_root == orchestrator.workspace_root

    def test_get_handler(self, orchestrator):
        """Test getting handlers for different action types."""
        handlers = ActionHandlers(orchestrator)

        # Test known action types
        message_handler = handlers.get_handler(ActionType.MESSAGE)
        assert message_handler == handlers._handle_message

        read_handler = handlers.get_handler(ActionType.READ_FILE)
        assert read_handler == handlers._handle_read_file

        # Test unknown action type
        unknown_handler = handlers.get_handler(None)
        assert unknown_handler is None

    def test_message_handling(self, orchestrator):
        """Test message action handling."""
        handlers = ActionHandlers(orchestrator)
        action = AgentAction(type=ActionType.MESSAGE, text="Hello")

        result = handlers._handle_message(action)
        assert result.success is True
        assert result.output == "Hello"
        assert result.type == "message"

    def test_finish_handling(self, orchestrator):
        """Test finish action handling."""
        handlers = ActionHandlers(orchestrator)
        action = AgentAction(type=ActionType.FINISH, text="Done")

        result = handlers._handle_finish(action)
        assert result.success is True
        assert result.output == "Done"
        assert result.type == "finish"


class TestValidationEngine:
    """Test the validation engine component."""

    def test_validation_engine_initialization(self, orchestrator):
        """Test that validation engine initializes correctly."""
        engine = ValidationEngine(orchestrator)

        assert engine.orchestrator == orchestrator
        assert engine.workspace_root == orchestrator.workspace_root

    def test_pre_execution_validation_valid_action(self, orchestrator):
        """Test pre-execution validation for valid actions."""
        engine = ValidationEngine(orchestrator)

        # Test valid message action
        action = AgentAction(type=ActionType.MESSAGE, text="Valid")
        result = engine.validate_pre_execution(action)

        assert result.success is True
        assert "validation passed" in result.output.lower()

    def test_pre_execution_validation_invalid_action(self, orchestrator):
        """Test pre-execution validation for invalid actions."""
        engine = ValidationEngine(orchestrator)

        # Test action with no type
        action = AgentAction(type=None, text="Invalid")
        result = engine.validate_pre_execution(action)

        assert result.success is False
        assert "not specified" in result.error

    def test_patch_validation(self, orchestrator):
        """Test patch validation functionality."""
        engine = ValidationEngine(orchestrator)

        # Test validation with empty paths (should pass)
        py_paths = set()
        original_states = {}
        backups = {}

        is_valid, output = engine.validate_patch_syntax(py_paths, original_states, backups)
        assert is_valid is True
        assert output == ""


class TestRollbackManager:
    """Test the rollback manager component."""

    def test_rollback_manager_initialization(self, orchestrator):
        """Test that rollback manager initializes correctly."""
        manager = RollbackManager(orchestrator)

        assert manager.orchestrator == orchestrator
        assert manager.workspace_root == orchestrator.workspace_root
        assert len(manager.rollback_checkpoints) > 0  # Should have initial checkpoint
        assert len(manager.rollback_triggers) > 0

    def test_checkpoint_creation(self, orchestrator):
        """Test checkpoint creation."""
        manager = RollbackManager(orchestrator)

        checkpoint_id = manager._create_checkpoint("test_checkpoint", "Test description")
        assert checkpoint_id is not None
        assert len(checkpoint_id) == 16  # Should be 16 character hash

        # Check that checkpoint was stored
        assert checkpoint_id in manager.rollback_checkpoints
        checkpoint_info = manager.rollback_checkpoints[checkpoint_id]
        assert checkpoint_info["name"] == "test_checkpoint"
        assert checkpoint_info["description"] == "Test description"

    def test_workspace_snapshot(self, orchestrator):
        """Test workspace file snapshotting."""
        manager = RollbackManager(orchestrator)

        snapshot = manager._snapshot_workspace_files()
        assert isinstance(snapshot, dict)

        # Should contain some files (at minimum __init__.py files)
        assert len(snapshot) > 0


class TestCodeAnalyzer:
    """Test the code analyzer component."""

    def test_code_analyzer_initialization(self, orchestrator):
        """Test that code analyzer initializes correctly."""
        analyzer = CodeAnalyzer(orchestrator)

        assert analyzer.orchestrator == orchestrator

    def test_impact_analysis_empty_diff(self, orchestrator):
        """Test impact analysis with empty diff."""
        analyzer = CodeAnalyzer(orchestrator)

        action = AgentAction(type=ActionType.SELF_MODIFY, diff="")
        is_safe, message = analyzer.analyze_impact_before_apply(action)

        assert is_safe is False
        assert "empty diff" in message.lower()

    def test_complexity_analysis(self, orchestrator):
        """Test code complexity analysis."""
        analyzer = CodeAnalyzer(orchestrator)

        # Test with simple Python code
        simple_code = "def hello(): return 'world'"
        metrics = analyzer.calculate_cyclomatic_complexity(simple_code)

        assert "total_complexity" in metrics
        assert "function_count" in metrics
        assert metrics["function_count"] == 1.0

    def test_python_code_extraction(self, orchestrator):
        """Test Python code extraction from diff."""
        analyzer = CodeAnalyzer(orchestrator)

        # Test diff with Python file
        diff_content = """+++ b/test.py
+def test_function():
+    return "hello"
"""
        extracted = analyzer._extract_python_code_from_diff(diff_content)
        assert "def test_function" in extracted
        assert 'return "hello"' in extracted


class TestSafetyMonitor:
    """Test the safety monitor component."""

    def test_safety_monitor_initialization(self, orchestrator):
        """Test that safety monitor initializes correctly."""
        monitor = SafetyMonitor(orchestrator)

        assert monitor.orchestrator == orchestrator
        assert monitor.config == orchestrator.config

    def test_resource_limits(self, orchestrator):
        """Test resource limits functionality."""
        monitor = SafetyMonitor(orchestrator)

        limits = monitor.get_resource_limits()
        assert "memory_limit_mb" in limits
        assert "command_timeout_sec" in limits

    def test_command_safety_validation(self, orchestrator):
        """Test command safety validation."""
        monitor = SafetyMonitor(orchestrator)

        # Test safe command
        safe_cmd = ["echo", "hello"]
        assert monitor.validate_command_safety(safe_cmd) is True

        # Test unsafe command (sudo)
        unsafe_cmd = ["sudo", "rm", "-rf", "/"]
        assert monitor.validate_command_safety(unsafe_cmd) is False

    def test_environment_restrictions(self, orchestrator):
        """Test environment restriction functionality."""
        monitor = SafetyMonitor(orchestrator)

        env = monitor.build_restricted_environment()
        assert "PATH" in env
        assert "HOME" in env
        assert "SHELL" in env

        # Check that PATH is restricted
        assert "/usr/local/bin:/usr/bin:/bin" == env["PATH"]


class TestPerformanceMonitor:
    """Test the performance monitor component."""

    def test_performance_monitor_initialization(self, orchestrator):
        """Test that performance monitor initializes correctly."""
        monitor = PerformanceMonitor(orchestrator)

        assert monitor.orchestrator == orchestrator
        assert "execution_times" in monitor.metrics
        assert "action_counts" in monitor.metrics

    def test_execution_monitoring(self, orchestrator):
        """Test execution monitoring functionality."""
        monitor = PerformanceMonitor(orchestrator)

        # Test monitoring context
        with monitor.monitor_execution(AgentAction(type=ActionType.MESSAGE)):
            # Simulate some work
            import time
            time.sleep(0.01)

        # Check that metrics were recorded
        assert monitor.metrics["total_executions"] > 0
        assert len(monitor.metrics["execution_times"]) > 0

    def test_performance_summary(self, orchestrator):
        """Test performance summary generation."""
        monitor = PerformanceMonitor(orchestrator)

        # Record some mock executions (simulate proper execution flow)
        monitor._record_execution_start("test_action")
        monitor._record_execution_complete("test_action", 0.1)
        monitor._record_execution_start("test_action")
        monitor._record_execution_complete("test_action", 0.2)

        summary = monitor.get_performance_summary()
        assert "summary" in summary
        assert "by_action_type" in summary
        assert summary["summary"]["total_executions"] == 2


class TestIntegration:
    """Integration tests for the modular execution system."""

    def test_full_execution_pipeline(self, test_config):
        """Test the full execution pipeline with all components."""
        orchestrator = ExecutionOrchestrator(test_config)

        # Test message action through full pipeline
        action = AgentAction(type=ActionType.MESSAGE, text="Integration test")
        result = orchestrator.execute(action)

        assert result.success is True
        assert result.output == "Integration test"

        # Check that performance was monitored
        status = orchestrator.get_component_status()
        assert status["performance_monitor"]["metrics"]["total_executions"] > 0

    def test_component_interaction(self, test_config):
        """Test that components interact correctly."""
        orchestrator = ExecutionOrchestrator(test_config)

        # Execute an action that triggers multiple components
        action = AgentAction(type=ActionType.MESSAGE, text="Component interaction test")
        result = orchestrator.execute(action)

        # Check that all components report activity
        status = orchestrator.get_component_status()

        # Performance monitor should have recorded the execution
        perf_status = status["performance_monitor"]
        assert perf_status["metrics"]["total_executions"] > 0

        # Validation engine should have processed the action
        val_status = status["validation_engine"]
        assert "validation_types" in val_status

    def test_error_handling_integration(self, test_config):
        """Test error handling across the modular system."""
        orchestrator = ExecutionOrchestrator(test_config)

        # Test with invalid action type
        action = AgentAction(type=None, text="Error test")
        result = orchestrator.execute(action)

        # Should handle gracefully
        assert result.success is False
        assert "not specified" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])