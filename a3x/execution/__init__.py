"""Modular execution system for A3X agent.

This module provides a refactored, maintainable execution architecture
that replaces the monolithic executor.py with focused, single-responsibility
components for better testability, maintainability, and evolution support.
"""

from .core import ExecutionOrchestrator
from .actions import ActionHandlers

__all__ = ["ExecutionOrchestrator", "ActionHandlers"]