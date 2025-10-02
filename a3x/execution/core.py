"""Core execution orchestration for A3X agent.

This module contains the main execution orchestrator that coordinates
between different execution components and maintains the overall execution flow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from ..actions import AgentAction, Observation
from ..config import AgentConfig
from ..change_log import ChangeLogger
from ..constants import (
    DAYS_BEFORE_ARCHIVE,
    FAILURE_RATE_THRESHOLD,
    MAX_DIFF_COMPLEXITY_SCORE,
    MAX_DIFF_LINES,
    MAX_FUNCTION_COMPLEXITY,
    MEMORY_LIMIT_MB,
    SUBPROCESS_TIMEOUT,
    TEST_FAILURE_RATE_THRESHOLD,
)
from ..patch import PatchManager

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .actions import ActionHandlers
    from .validation import ValidationEngine
    from .rollback import RollbackManager
    from .analysis import CodeAnalyzer
    from .safety import SafetyMonitor
    from .monitoring import PerformanceMonitor


class ExecutionOrchestrator:
    """Main execution orchestrator that coordinates all execution components.

    This class replaces the monolithic ActionExecutor and delegates specific
    responsibilities to focused, single-purpose components for better
    maintainability and testability.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the ExecutionOrchestrator with configuration.

        Args:
            config (AgentConfig): The agent configuration containing workspace,
                limits, policies, and audit settings.
        """
        self.config = config
        self.workspace_root = Path(config.workspace.root).resolve()

        # Initialize core components
        self.patch_manager = PatchManager(self.workspace_root)
        self.change_logger = ChangeLogger(
            self.workspace_root,
            enable_file_log=config.audit.enable_file_log,
            file_dir=config.audit.file_dir,
            enable_git_commit=config.audit.enable_git_commit,
            commit_prefix=config.audit.commit_prefix,
        )

        # Initialize modular components (lazy initialization to avoid circular imports)
        self._action_handlers = None
        self._validation_engine = None
        self._rollback_manager = None
        self._code_analyzer = None
        self._safety_monitor = None
        self._performance_monitor = None

    def execute(self, action: AgentAction) -> Observation:
        """Execute the given agent action and return an observation.

        Dispatches to specific handlers based on action type, with full
        safety checking, validation, and monitoring.

        Args:
            action (AgentAction): The action to execute, containing type, command,
                path, content, or diff as appropriate.

        Returns:
            Observation: Result of execution with success flag, output, error message,
                return code (for commands), duration, and type.
        """
        # Pre-execution validation and safety checks
        validation_result = self.validation_engine.validate_pre_execution(action)
        if not validation_result.success:
            return validation_result

        # Execute the action through the appropriate handler
        handler = self.action_handlers.get_handler(action.type)
        if handler is None:
            return Observation(
                success=False,
                output="",
                error=f"Unsupported action type: {action.type}"
            )

        # Execute with monitoring
        with self.performance_monitor.monitor_execution(action):
            result = handler(action)

        # Post-execution validation and rollback checks
        self.validation_engine.validate_post_execution(action, result)

        # Check if rollback should be triggered based on result
        self.rollback_manager.check_rollback_triggers(result)

        return result

    def get_workspace_path(self, path: str) -> Path:
        """Resolve a workspace-relative path to absolute path.

        Args:
            path (str): The relative path to resolve.

        Returns:
            Path: The absolute path within the workspace.
        """
        candidate = (
            (self.workspace_root / path).resolve()
            if not Path(path).is_absolute()
            else Path(path).resolve()
        )

        def _is_within(base: Path, target: Path) -> bool:
            try:
                target.relative_to(base)
                return True
            except ValueError:
                return False

        if not self.config.workspace.allow_outside_root:
            if not _is_within(self.workspace_root, candidate) and not _is_within(
                Path("/tmp/a3x_sandbox"), candidate
            ):
                raise PermissionError(
                    f"Access denied outside workspace: {candidate}"
                )

        return candidate

    @property
    def action_handlers(self):
        """Get the action handlers component."""
        if self._action_handlers is None:
            from .actions import ActionHandlers
            self._action_handlers = ActionHandlers(self)
        return self._action_handlers

    @property
    def validation_engine(self):
        """Get the validation engine component."""
        if self._validation_engine is None:
            from .validation import ValidationEngine
            self._validation_engine = ValidationEngine(self)
        return self._validation_engine

    @property
    def rollback_manager(self):
        """Get the rollback manager component."""
        if self._rollback_manager is None:
            from .rollback import RollbackManager
            self._rollback_manager = RollbackManager(self)
        return self._rollback_manager

    @property
    def code_analyzer(self):
        """Get the code analyzer component."""
        if self._code_analyzer is None:
            from .analysis import CodeAnalyzer
            self._code_analyzer = CodeAnalyzer(self)
        return self._code_analyzer

    @property
    def safety_monitor(self):
        """Get the safety monitor component."""
        if self._safety_monitor is None:
            from .safety import SafetyMonitor
            self._safety_monitor = SafetyMonitor(self)
        return self._safety_monitor

    @property
    def performance_monitor(self):
        """Get the performance monitor component."""
        if self._performance_monitor is None:
            from .monitoring import PerformanceMonitor
            self._performance_monitor = PerformanceMonitor(self)
        return self._performance_monitor

    def get_component_status(self) -> Dict[str, Any]:
        """Get status information from all execution components.

        Returns:
            Dict[str, Any]: Status information including metrics, health checks,
                and configuration state from all components.
        """
        return {
            "action_handlers": self.action_handlers.get_status(),
            "validation_engine": self.validation_engine.get_status(),
            "rollback_manager": self.rollback_manager.get_status(),
            "code_analyzer": self.code_analyzer.get_status(),
            "safety_monitor": self.safety_monitor.get_status(),
            "performance_monitor": self.performance_monitor.get_status(),
        }