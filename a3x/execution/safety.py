"""Safety monitoring for A3X execution system.

This module contains safety monitoring functionality including command
safety checks, resource limits, and sandboxing that was previously
embedded in the monolithic ActionExecutor class.
"""

from __future__ import annotations

import os
import re
import resource
from typing import Dict, Any, List

from ..actions import AgentAction

class SafetyMonitor:
    """Safety monitor for command execution and resource management.

    This class handles safety checks, resource limits, and sandboxing
    to ensure secure execution of actions.
    """

    def __init__(self, orchestrator) -> None:
        """Initialize the safety monitor.

        Args:
            orchestrator (ExecutionOrchestrator): The main execution orchestrator.
        """
        self.orchestrator = orchestrator
        self.config = orchestrator.config

    def get_status(self) -> Dict[str, Any]:
        """Get status information about the safety monitor.

        Returns:
            Dict[str, Any]: Status information including safety capabilities.
        """
        return {
            "safety_checks": [
                "command_validation", "resource_limits", "environment_restriction",
                "network_policy", "privilege_escalation"
            ],
            "resource_limits": {
                "memory_mb": 100,  # Default memory limit
                "timeout_sec": self.config.limits.command_timeout,
            },
            "policies": {
                "allow_network": self.config.policies.allow_network,
                "deny_commands": self.config.policies.deny_commands,
            }
        }

    def validate_command_safety(self, command: List[str]) -> bool:
        """Validate that a command is safe to execute.

        Args:
            command (List[str]): The command to validate.

        Returns:
            bool: True if command is safe, False otherwise.
        """
        # Check against policy
        if not self._command_allowed_by_policy(command):
            return False

        # Check for unsafe patterns
        if not self._is_safe_command(command):
            return False

        return True

    def get_resource_limits(self) -> Dict[str, int]:
        """Get current resource limits.

        Returns:
            Dict[str, int]: Resource limits including memory and timeout.
        """
        return {
            "memory_limit_mb": 100,  # Default memory limit
            "command_timeout_sec": self.config.limits.command_timeout,
        }

    def apply_resource_limits(self) -> None:
        """Apply resource limits for command execution."""
        # Set memory limit (default 100MB if not configured)
        memory_limit = 100 * 1024 * 1024  # 100MB default
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        except (ValueError, OSError) as e:
            # Log warning but don't fail - limits might not be settable on all systems
            print(f"Warning: Could not set memory limit: {e}")

    def build_restricted_environment(self) -> Dict[str, str]:
        """Build restricted environment for command execution.

        Returns:
            Dict[str, str]: Restricted environment variables.
        """
        env = self._build_base_environment()

        # Remove dangerous variables
        dangerous_vars = ["PATH", "LD_PRELOAD", "LD_LIBRARY_PATH"]
        for var in dangerous_vars:
            env.pop(var, None)

        # Set restricted PATH
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"

        # Set workspace as home
        env["HOME"] = str(self.orchestrator.workspace_root)
        env["SHELL"] = "/bin/sh"

        return env

    def _command_allowed_by_policy(self, command: List[str]) -> bool:
        """Check if command is allowed by current policies."""
        joined = " ".join(command)
        for pattern in self.config.policies.deny_commands:
            if pattern in joined:
                return False
        return True

    def _is_safe_command(self, command: List[str]) -> bool:
        """Check if command contains unsafe patterns."""
        joined = " ".join(command).lower()
        unsafe_patterns = [
            "sudo", "su", "rm -rf", "dd if=", "mkfs", "mount", "umount",
            "chmod +x", "chown", "passwd", "usermod", "userdel",
        ]

        # Check for privilege escalation
        if any(unsafe in joined for unsafe in unsafe_patterns):
            return False

        # Check network restrictions
        if not self.config.policies.allow_network:
            network_commands = ["curl", "wget", "ping", "netcat", "nc", "ssh", "scp", "rsync"]
            if any(net_cmd in joined for net_cmd in network_commands):
                return False

        return True

    def _build_base_environment(self) -> Dict[str, str]:
        """Build base environment with policy restrictions."""
        env = os.environ.copy()

        # Apply network restrictions
        if not self.config.policies.allow_network:
            env.setdefault("NO_NETWORK", "1")
            # Could also set http_proxy to empty or localhost to block external access

        return env

    def check_file_access_safety(self, file_path: str) -> bool:
        """Check if file access is safe.

        Args:
            file_path (str): Path to check.

        Returns:
            bool: True if access is safe.
        """
        try:
            target_path = self.orchestrator.get_workspace_path(file_path)
            # Additional safety checks could be added here
            return True
        except PermissionError:
            return False

    def validate_action_safety(self, action: AgentAction) -> bool:
        """Validate overall safety of an action.

        Args:
            action (AgentAction): The action to validate.

        Returns:
            bool: True if action is safe.
        """
        # Check action type specific safety
        if action.type.name == "RUN_COMMAND":
            return self.validate_command_safety(action.command or [])
        elif action.type.name in ["READ_FILE", "WRITE_FILE"]:
            return self.check_file_access_safety(action.path or "")
        elif action.type.name in ["APPLY_PATCH", "SELF_MODIFY"]:
            return self.validate_patch_safety(action.diff or "")
        else:
            # Other action types are generally safe
            return True

    def validate_patch_safety(self, diff: str) -> bool:
        """Validate safety of a patch.

        Args:
            diff (str): The patch diff to validate.

        Returns:
            bool: True if patch is safe.
        """
        # Check for dangerous patterns in the patch
        dangerous_patterns = [
            r"\+.*sudo",
            r"\+.*chmod.*777",
            r"\+.*allow_network.*=.*True",
            r"-.*deny_commands.*=\[\]",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, diff):
                return False

        return True

    def get_safety_report(self) -> Dict[str, Any]:
        """Get comprehensive safety report.

        Returns:
            Dict[str, Any]: Safety status and configuration.
        """
        return {
            "resource_limits": self.get_resource_limits(),
            "environment_restrictions": self.build_restricted_environment(),
            "policy_status": {
                "network_allowed": self.config.policies.allow_network,
                "denied_commands": self.config.policies.deny_commands,
            },
            "safety_checks_enabled": [
                "command_validation",
                "resource_limits",
                "environment_restriction",
                "file_access_control",
                "patch_validation"
            ]
        }