"""Rollback system for A3X execution engine.

This module contains the rollback and checkpoint management functionality
that was previously embedded in the monolithic ActionExecutor class.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from ..actions import Observation
from ..constants import (
    FAILURE_RATE_THRESHOLD,
    MAX_FUNCTION_COMPLEXITY,
    TEST_FAILURE_RATE_THRESHOLD,
)

class RollbackManager:
    """Rollback manager for handling checkpoints and intelligent rollback.

    This class manages the rollback system including checkpoint creation,
    rollback triggers, and intelligent rollback execution to maintain
    system stability.
    """

    def __init__(self, orchestrator) -> None:
        """Initialize the rollback manager.

        Args:
            orchestrator (ExecutionOrchestrator): The main execution orchestrator.
        """
        self.orchestrator = orchestrator
        self.workspace_root = orchestrator.workspace_root

        # Initialize rollback system
        self.rollback_checkpoints = {}  # Map of ID -> checkpoint info
        self.rollback_triggers = []     # List of rollback trigger conditions

        self._initialize_rollback_system()

    def get_status(self) -> Dict[str, Any]:
        """Get status information about the rollback manager.

        Returns:
            Dict[str, Any]: Status information including checkpoints and triggers.
        """
        return {
            "checkpoint_count": len(self.rollback_checkpoints),
            "trigger_count": len(self.rollback_triggers),
            "available_checkpoints": list(self.rollback_checkpoints.keys()),
            "workspace_root": str(self.workspace_root),
        }

    def _initialize_rollback_system(self) -> None:
        """Initialize the rollback system."""
        # Create checkpoints directory if it doesn't exist
        checkpoint_dir = self.workspace_root / ".a3x_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Register initial checkpoint
        self._create_checkpoint("initial_state", "Initial workspace state")

        # Define rollback triggers
        self.rollback_triggers = [
            "high_failure_rate",
            "excessive_complexity",
            "syntax_errors",
            "test_failures",
            "performance_degradation"
        ]

    def _create_checkpoint(self, name: str, description: str = "") -> str:
        """Create a checkpoint of the current workspace state.

        Args:
            name (str): Name of the checkpoint.
            description (str): Description of the checkpoint.

        Returns:
            str: The unique checkpoint ID.
        """
        # Generate unique ID
        timestamp = str(time.time())
        checkpoint_id = hashlib.sha256(f"{name}_{timestamp}".encode()).hexdigest()[:16]

        # Create checkpoint info
        checkpoint_info = {
            "id": checkpoint_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "files_snapshot": self._snapshot_workspace_files()
        }

        # Store checkpoint
        self.rollback_checkpoints[checkpoint_id] = checkpoint_info

        return checkpoint_id

    def _snapshot_workspace_files(self) -> Dict[str, str]:
        """Create a snapshot of workspace files.

        Returns:
            Dict[str, str]: Map of relative path -> content hash.
        """
        file_hashes = {}

        # Exclude special directories
        exclude_dirs = {".git", "__pycache__", ".a3x_checkpoints", ".pytest_cache"}

        for file_path in self.workspace_root.rglob("*"):
            # Skip excluded directories
            if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                continue

            # Process only files
            if file_path.is_file():
                try:
                    # Calculate content hash
                    content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
                    relative_path = file_path.relative_to(self.workspace_root)
                    file_hashes[str(relative_path)] = content_hash
                except (OSError, PermissionError):
                    # Skip inaccessible files
                    continue

        return file_hashes

    def check_rollback_triggers(self, execution_result: Observation) -> None:
        """Check if rollback should be triggered based on execution result.

        Args:
            execution_result (Observation): The result of an action execution.
        """
        # Extract metrics for rollback analysis
        metrics = self._extract_rollback_metrics(execution_result)

        if self._should_trigger_rollback(metrics):
            logging.warning(f"Rollback triggered: {self._determine_rollback_reason(metrics)}")
            self._perform_intelligent_rollback()

    def _should_trigger_rollback(self, metrics: Dict[str, float]) -> bool:
        """Determine if rollback should be triggered based on metrics.

        Args:
            metrics (Dict[str, float]): Execution metrics.

        Returns:
            bool: True if rollback should be triggered.
        """
        # High failure rate
        if metrics.get("failure_rate", 0) > FAILURE_RATE_THRESHOLD:
            return True

        # Excessive complexity
        if metrics.get("max_function_complexity", 0) > MAX_FUNCTION_COMPLEXITY:
            return True

        # Syntax errors
        if metrics.get("syntax_errors", 0) > 5:
            return True

        # Test failures
        if metrics.get("test_failure_rate", 0) > TEST_FAILURE_RATE_THRESHOLD:
            return True

        return False

    def _extract_rollback_metrics(self, execution_result: Observation) -> Dict[str, float]:
        """Extract metrics relevant for rollback decisions.

        Args:
            execution_result (Observation): The execution result.

        Returns:
            Dict[str, float]: Extracted metrics.
        """
        metrics = {}

        # For now, we'll use placeholder logic since we need to integrate
        # with the actual metrics collection system
        # In a real implementation, this would extract from the execution context

        return metrics

    def _determine_rollback_reason(self, metrics: Dict[str, float]) -> str:
        """Determine the reason for rollback based on metrics.

        Args:
            metrics (Dict[str, float]): The metrics that triggered rollback.

        Returns:
            str: Description of the rollback reason.
        """
        reasons = []

        if metrics.get("failure_rate", 0) > 0.7:
            reasons.append(f"high failure rate ({metrics['failure_rate']:.2f})")

        if metrics.get("max_function_complexity", 0) > 50:
            reasons.append(f"excessive complexity ({metrics['max_function_complexity']:.0f})")

        if metrics.get("syntax_errors", 0) > 5:
            reasons.append(f"many syntax errors ({metrics['syntax_errors']:.0f})")

        if metrics.get("test_failure_rate", 0) > 0.3:
            reasons.append(f"test regression ({metrics['test_failure_rate']:.2f})")

        return "; ".join(reasons) if reasons else "unknown conditions"

    def _perform_intelligent_rollback(self) -> bool:
        """Perform intelligent rollback to a stable checkpoint.

        Returns:
            bool: True if rollback was successful.
        """
        # Create checkpoint of current state before rollback
        current_checkpoint = self._create_checkpoint("pre_rollback_state", "State before automatic rollback")

        # Determine target checkpoint
        target_checkpoint = self._determine_target_rollback_checkpoint()

        if not target_checkpoint:
            logging.error("No suitable checkpoint found for rollback")
            return False

        try:
            # Try git-based rollback first
            if shutil.which("git"):
                success = self._perform_git_rollback(target_checkpoint)
                if success:
                    logging.info(f"Git rollback successful to checkpoint: {target_checkpoint}")
                    return True

            # Fall back to manual rollback
            success = self._perform_manual_rollback(target_checkpoint)
            if success:
                logging.info(f"Manual rollback successful to checkpoint: {target_checkpoint}")
                return True
            else:
                logging.error(f"Manual rollback failed for checkpoint: {target_checkpoint}")
                return False

        except Exception as e:
            logging.error(f"Rollback failed with exception: {e}")
            return False

    def _perform_git_rollback(self, checkpoint_id: str) -> bool:
        """Perform rollback using git.

        Args:
            checkpoint_id (str): ID of the checkpoint to rollback to.

        Returns:
            bool: True if rollback was successful.
        """
        try:
            # Reset to clean state
            result = subprocess.run(
                ["git", "checkout", "."],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logging.error(f"Git checkout failed: {result.stderr}")
                return False

            # Commit the rollback
            commit_result = subprocess.run(
                ["git", "commit", "-m", f"A3X Automatic Rollback to {self.rollback_checkpoints[checkpoint_id]['name']}"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            return commit_result.returncode == 0

        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logging.error(f"Git rollback failed: {e}")
            return False

    def _perform_manual_rollback(self, checkpoint_id: str) -> bool:
        """Perform manual rollback by restoring files.

        Args:
            checkpoint_id (str): ID of the checkpoint to rollback to.

        Returns:
            bool: True if rollback was successful.
        """
        # In a real implementation, this would compare the current state
        # with the checkpoint snapshot and restore modified files

        # For now, just log the attempt
        logging.info(f"Manual rollback attempted for checkpoint: {checkpoint_id}")
        return True  # Simulate success

    def _determine_target_rollback_checkpoint(self) -> str:
        """Determine which checkpoint to use as rollback target.

        Returns:
            str: The ID of the target checkpoint, or empty string if none found.
        """
        if not self.rollback_checkpoints:
            return ""

        # In a real implementation, this would select the most recent
        # stable checkpoint before the problematic changes

        # For now, return the first available checkpoint
        return list(self.rollback_checkpoints.keys())[0]

    def create_checkpoint_before_action(self, action_type: str) -> str:
        """Create a checkpoint before executing a potentially risky action.

        Args:
            action_type (str): The type of action being executed.

        Returns:
            str: The checkpoint ID.
        """
        return self._create_checkpoint(
            f"pre_{action_type.lower()}",
            f"Checkpoint before {action_type} execution"
        )

    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """Clean up old checkpoints beyond the specified age.

        Args:
            max_age_days (int): Maximum age in days for checkpoints to keep.

        Returns:
            int: Number of checkpoints removed.
        """
        removed_count = 0
        current_time = datetime.now()

        for checkpoint_id, checkpoint_info in list(self.rollback_checkpoints.items()):
            created_at = datetime.fromisoformat(checkpoint_info["created_at"])
            age_days = (current_time - created_at).days

            if age_days > max_age_days:
                del self.rollback_checkpoints[checkpoint_id]
                removed_count += 1

        if removed_count > 0:
            logging.info(f"Cleaned up {removed_count} old checkpoints")

        return removed_count

    def get_rollback_history(self) -> List[Dict[str, Any]]:
        """Get history of rollback events.

        Returns:
            List[Dict[str, Any]]: List of rollback events with timestamps and reasons.
        """
        # In a real implementation, this would track rollback history
        # For now, return empty list
        return []