"""Refactored execution system for A3X agent.

This module provides a facade over the new modular execution system
while maintaining backward compatibility with the original ActionExecutor API.
"""

from __future__ import annotations

from .actions import AgentAction, Observation
from .config import AgentConfig
from .execution import ExecutionOrchestrator


class ActionExecutor:
    """Facade for the refactored modular execution system.

    This class maintains backward compatibility with the original ActionExecutor API
    while delegating all functionality to the new modular execution system.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the ActionExecutor with the new modular system.

        Args:
            config (AgentConfig): The agent configuration containing workspace,
                limits, policies, and audit settings.
        """
        self.config = config
        self.orchestrator = ExecutionOrchestrator(config)

    def execute(self, action: AgentAction) -> Observation:
        """Execute the given agent action using the modular system.

        Args:
            action (AgentAction): The action to execute.

        Returns:
            Observation: Result of execution with success flag, output, error message,
                return code (for commands), duration, and type.
        """
        return self.orchestrator.execute(action)

    # Backward compatibility properties for any code that might access internal state
    @property
    def workspace_root(self) -> Path:
        """Get the workspace root path."""
        return self.orchestrator.workspace_root

    @property
    def patch_manager(self):
        """Get the patch manager."""
        return self.orchestrator.patch_manager

    @property
    def change_logger(self):
        """Get the change logger."""
        return self.orchestrator.change_logger

    # Component status access for backward compatibility
    def get_component_status(self) -> Dict[str, Any]:
        """Get status from all execution components."""
        return self.orchestrator.get_component_status()

