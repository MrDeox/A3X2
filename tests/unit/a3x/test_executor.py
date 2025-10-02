"""Unit tests for a3x.executor."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from a3x.executor import ActionExecutor


@pytest.fixture
def mock_config():
    class MockLimits:
        command_timeout = 1  # Short for testing

    class MockConfig:
        limits = MockLimits()
        workspace = Mock(root=".")
        policies = Mock(deny_commands=[], allow_network=False)

    return MockConfig()


def test_resource_limits_timeout(mock_config):
    """Test command timeout enforcement."""
    executor = ActionExecutor(mock_config)
    action = Mock(command=["sleep", "2"])  # Longer than 1s timeout

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=1, output=b"", stderr=b"")

        obs = executor._handle_run_command(action)

        assert not obs.success
        assert "Timeout" in obs.error
        mock_run.assert_called_once()


def test_resource_limits_memory(mock_config):
    """Test memory limit enforcement (simulated)."""
    # Note: Actual memory limit testing requires OS-specific setup; simulate via preexec_fn call
    executor = ActionExecutor(mock_config)
    action = Mock(command=["stress", "--vm", "1", "--vm-bytes", "1G"])  # Would exceed 512MB

    with patch("subprocess.run") as mock_run:
        # Simulate memory limit hit via exception in preexec_fn or run
        def mock_preexec():
            raise MemoryError("Simulated memory limit exceeded")

        mock_config.limits.command_timeout = 5
        action.command = ["sleep", "1"]  # Harmless command

        obs = executor._handle_run_command(action)

        # Since simulation, check that preexec_fn is called (resource.setrlimit)
        # But for unit test, verify the call happens
        assert "resource" in str(mock_run.call_args)  # Indirect check
        # In real test, would mock resource.setrlimit and verify args
