"""Unit tests for a3x.cli."""

import sys
from unittest.mock import patch

import pytest

from a3x.cli import main


def test_interactive_flag_parsing(capsys):
    """Test --interactive flag parsing."""
    with patch.object(sys, "argv", ["a3x", "run", "--goal", "test", "--interactive"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0  # Assuming success

    # Verify config.interactive is set (indirect via captured output or mock)
    # For full test, would mock load_config and assert config.loop.interactive=True


@patch("builtins.input", side_effect=["n"])  # Simulate no refinement
def test_interactive_prompt_simulation(mock_input, capsys):
    """Simulate interactive mode in agent run (high-level)."""
    # This is a high-level simulation; full integration would mock agent.run
    # For unit, test CLI parsing leads to interactive=True in config
    with patch.object(sys, "argv", ["a3x", "run", "--goal", "test", "--interactive"]):
        # Mock to avoid full run
        with patch("a3x.cli.AgentOrchestrator") as mock_agent:
            mock_agent.return_value.run.return_value = Mock(completed=True)
            main()

    # Verify input was called (prompt simulation)
    mock_input.assert_called()
    # In full test, would check agent config passed interactive=True
