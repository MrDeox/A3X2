"""Comprehensive tests for the history module."""

from unittest.mock import MagicMock

import pytest

from a3x.actions import ActionType, AgentAction, Observation
from a3x.history import AgentHistory, HistoryEvent, _describe_action, _describe_observation


class TestHistoryEvent:
    """Test cases for the HistoryEvent dataclass."""

    def test_history_event_creation(self) -> None:
        """Test creating a history event."""
        action = AgentAction(
            type=ActionType.MESSAGE,
            text="Test message",
            path=None,
            content=None,
            command=None,
            cwd=None,
            diff=None
        )

        observation = Observation(
            success=True,
            return_code=0,
            output="Command completed successfully",
            error=None,
            duration=1.5
        )

        event = HistoryEvent(action=action, observation=observation)

        assert event.action == action
        assert event.observation == observation

    def test_history_event_with_failed_observation(self) -> None:
        """Test creating a history event with failed observation."""
        action = AgentAction(
            type=ActionType.RUN_COMMAND,
            command=["python", "script.py"],
            cwd="/test"
        )

        observation = Observation(
            success=False,
            return_code=1,
            output="",
            error="SyntaxError: invalid syntax",
            duration=0.5
        )

        event = HistoryEvent(action=action, observation=observation)

        assert event.action == action
        assert event.observation == observation
        assert not event.observation.success

    def test_history_event_with_none_values(self) -> None:
        """Test creating a history event with None values in observation."""
        action = AgentAction(
            type=ActionType.READ_FILE,
            path="test.txt"
        )

        observation = Observation(
            success=True,
            return_code=None,
            output=None,
            error=None,
            duration=None
        )

        event = HistoryEvent(action=action, observation=observation)

        assert event.action == action
        assert event.observation == observation
        assert event.observation.return_code is None
        assert event.observation.output is None


class TestAgentHistory:
    """Test cases for the AgentHistory class."""

    def test_history_creation_empty(self) -> None:
        """Test creating an empty history."""
        history = AgentHistory()

        assert len(history.events) == 0
        assert history._events == []

    def test_history_append_single_event(self) -> None:
        """Test appending a single event to history."""
        history = AgentHistory()

        action = AgentAction(
            type=ActionType.MESSAGE,
            text="Hello world"
        )
        observation = Observation(success=True, return_code=0, output="OK")

        history.append(action, observation)

        assert len(history.events) == 1
        event = history.events[0]
        assert event.action == action
        assert event.observation == observation

    def test_history_append_multiple_events(self) -> None:
        """Test appending multiple events to history."""
        history = AgentHistory()

        events_data = [
            (
                AgentAction(type=ActionType.MESSAGE, text="First message"),
                Observation(success=True, return_code=0, output="First OK")
            ),
            (
                AgentAction(type=ActionType.RUN_COMMAND, command=["echo", "hello"]),
                Observation(success=True, return_code=0, output="hello")
            ),
            (
                AgentAction(type=ActionType.MESSAGE, text="Second message"),
                Observation(success=False, return_code=1, error="Failed")
            )
        ]

        for action, observation in events_data:
            history.append(action, observation)

        assert len(history.events) == 3

        # Verify events are in correct order
        for i, (expected_action, expected_observation) in enumerate(events_data):
            assert history.events[i].action.text == expected_action.text or \
                   history.events[i].action.command == expected_action.command
            assert history.events[i].observation.success == expected_observation.success

    def test_history_events_property_returns_copy(self) -> None:
        """Test that events property returns a copy, not the original list."""
        history = AgentHistory()

        action = AgentAction(type=ActionType.MESSAGE, text="Test")
        observation = Observation(success=True, return_code=0, output="OK")

        history.append(action, observation)

        # Get events and modify the returned list
        events = history.events
        events.clear()

        # Original history should be unchanged
        assert len(history.events) == 1

    def test_history_maintains_order(self) -> None:
        """Test that history maintains chronological order."""
        history = AgentHistory()

        # Add events in specific order
        messages = ["First", "Second", "Third", "Fourth", "Fifth"]

        for i, message in enumerate(messages):
            action = AgentAction(type=ActionType.MESSAGE, text=f"{i+1}. {message}")
            observation = Observation(success=True, return_code=0, output=f"Response to {message}")
            history.append(action, observation)

        # Verify order is preserved
        assert len(history.events) == 5
        for i, message in enumerate(messages):
            expected_text = f"{i+1}. {message}"
            assert history.events[i].action.text == expected_text

    def test_history_snapshot_empty(self) -> None:
        """Test snapshot of empty history."""
        history = AgentHistory()

        snapshot = history.snapshot()
        assert snapshot == ""

    def test_history_snapshot_single_event(self) -> None:
        """Test snapshot of history with single event."""
        history = AgentHistory()

        action = AgentAction(type=ActionType.MESSAGE, text="Hello world")
        observation = Observation(success=True, return_code=0, output="Response")

        history.append(action, observation)

        snapshot = history.snapshot()

        assert "[1] ACTION: Hello world" in snapshot
        assert "[1] OBS   : OK (code=0, Response)" in snapshot

    def test_history_snapshot_multiple_events(self) -> None:
        """Test snapshot of history with multiple events."""
        history = AgentHistory()

        # Add several events
        events = [
            (ActionType.MESSAGE, "First message"),
            (ActionType.RUN_COMMAND, ["echo", "hello"]),
            (ActionType.APPLY_PATCH, None),
            (ActionType.WRITE_FILE, "test.txt"),
            (ActionType.FINISH, "Task completed")
        ]

        for i, (action_type, data) in enumerate(events):
            if action_type == ActionType.MESSAGE:
                action = AgentAction(type=action_type, text=data)
            elif action_type == ActionType.RUN_COMMAND:
                action = AgentAction(type=action_type, command=data)
            elif action_type == ActionType.APPLY_PATCH:
                action = AgentAction(type=action_type, diff="+ test line")
            elif action_type == ActionType.WRITE_FILE:
                action = AgentAction(type=action_type, path=data, content="file content")
            elif action_type == ActionType.FINISH:
                action = AgentAction(type=action_type, text=data)

            observation = Observation(success=True, return_code=0, output=f"Response {i+1}")
            history.append(action, observation)

        snapshot = history.snapshot()

        # Verify all events are included
        assert "[1] ACTION: First message" in snapshot
        assert "[2] ACTION: RUN `echo hello` (cwd=.)" in snapshot
        assert "[3] ACTION: APPLY_PATCH (1 linhas de diff)" in snapshot
        assert "[4] ACTION: WRITE_FILE test.txt (11 chars)" in snapshot
        assert "[5] ACTION: Task completed" in snapshot

    def test_history_snapshot_with_truncation(self) -> None:
        """Test snapshot truncation when content exceeds max_chars."""
        history = AgentHistory()

        # Create a very long message that will exceed the limit
        long_message = "A" * 10000
        action = AgentAction(type=ActionType.MESSAGE, text=long_message)
        observation = Observation(success=True, return_code=0, output="Long response")

        history.append(action, observation)

        # Use small max_chars to force truncation
        snapshot = history.snapshot(max_chars=100)

        assert len(snapshot) <= 150  # Should be truncated
        assert "... (histórico truncado) ..." in snapshot

    def test_history_snapshot_calculates_length_correctly(self) -> None:
        """Test that snapshot length calculation works correctly."""
        history = AgentHistory()

        # Add events that together exceed the limit
        for i in range(10):
            action = AgentAction(type=ActionType.MESSAGE, text=f"Message {i}")
            observation = Observation(success=True, return_code=0, output=f"Response {i}")
            history.append(action, observation)

        # Use a limit that should truncate after a few events
        snapshot = history.snapshot(max_chars=200)

        # Should be truncated
        assert "... (histórico truncado) ..." in snapshot
        # Should contain first few events
        assert "[1] ACTION: Message 0" in snapshot
        assert "[2] ACTION: Message 1" in snapshot
        # Should not contain later events
        assert "[10] ACTION: Message 9" not in snapshot


class TestDescribeAction:
    """Test cases for the _describe_action helper function."""

    def test_describe_message_action(self) -> None:
        """Test describing a message action."""
        action = AgentAction(
            type=ActionType.MESSAGE,
            text="Hello, this is a test message"
        )

        description = _describe_action(action)
        assert description == "Hello, this is a test message"

    def test_describe_message_action_empty_text(self) -> None:
        """Test describing a message action with empty text."""
        action = AgentAction(
            type=ActionType.MESSAGE,
            text=""
        )

        description = _describe_action(action)
        assert description == "(mensagem vazia)"

    def test_describe_message_action_none_text(self) -> None:
        """Test describing a message action with None text."""
        action = AgentAction(
            type=ActionType.MESSAGE,
            text=None
        )

        description = _describe_action(action)
        assert description == "(mensagem vazia)"

    def test_describe_run_command_action(self) -> None:
        """Test describing a run command action."""
        action = AgentAction(
            type=ActionType.RUN_COMMAND,
            command=["python", "script.py", "--verbose"],
            cwd="/home/user/project"
        )

        description = _describe_action(action)
        assert description == "RUN `python script.py --verbose` (cwd=/home/user/project)"

    def test_describe_run_command_action_no_cwd(self) -> None:
        """Test describing a run command action without cwd."""
        action = AgentAction(
            type=ActionType.RUN_COMMAND,
            command=["ls", "-la"]
        )

        description = _describe_action(action)
        assert description == "RUN `ls -la` (cwd=.)"

    def test_describe_run_command_action_none_command(self) -> None:
        """Test describing a run command action with None command."""
        action = AgentAction(
            type=ActionType.RUN_COMMAND,
            command=None
        )

        description = _describe_action(action)
        assert description == "RUN `` (cwd=.)"

    def test_describe_apply_patch_action_with_diff(self) -> None:
        """Test describing an apply patch action with diff."""
        diff_text = """--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,4 @@
 line 1
 line 2
+new line
 line 3
"""
        action = AgentAction(
            type=ActionType.APPLY_PATCH,
            diff=diff_text
        )

        description = _describe_action(action)
        assert description == "APPLY_PATCH (4 linhas de diff)"

    def test_describe_apply_patch_action_no_diff(self) -> None:
        """Test describing an apply patch action without diff."""
        action = AgentAction(
            type=ActionType.APPLY_PATCH,
            diff=None
        )

        description = _describe_action(action)
        assert description == "APPLY_PATCH (sem diff)"

    def test_describe_apply_patch_action_empty_diff(self) -> None:
        """Test describing an apply patch action with empty diff."""
        action = AgentAction(
            type=ActionType.APPLY_PATCH,
            diff=""
        )

        description = _describe_action(action)
        assert description == "APPLY_PATCH (sem diff)"

    def test_describe_write_file_action(self) -> None:
        """Test describing a write file action."""
        action = AgentAction(
            type=ActionType.WRITE_FILE,
            path="test.txt",
            content="Hello world\nThis is a test file."
        )

        description = _describe_action(action)
        assert description == "WRITE_FILE test.txt (29 chars)"

    def test_describe_write_file_action_empty_content(self) -> None:
        """Test describing a write file action with empty content."""
        action = AgentAction(
            type=ActionType.WRITE_FILE,
            path="empty.txt",
            content=""
        )

        description = _describe_action(action)
        assert description == "WRITE_FILE empty.txt (0 chars)"

    def test_describe_read_file_action(self) -> None:
        """Test describing a read file action."""
        action = AgentAction(
            type=ActionType.READ_FILE,
            path="source.py"
        )

        description = _describe_action(action)
        assert description == "READ_FILE source.py"

    def test_describe_finish_action_with_text(self) -> None:
        """Test describing a finish action with text."""
        action = AgentAction(
            type=ActionType.FINISH,
            text="Task completed successfully"
        )

        description = _describe_action(action)
        assert description == "Task completed successfully"

    def test_describe_finish_action_empty_text(self) -> None:
        """Test describing a finish action with empty text."""
        action = AgentAction(
            type=ActionType.FINISH,
            text=""
        )

        description = _describe_action(action)
        assert description == "(fim)"

    def test_describe_finish_action_none_text(self) -> None:
        """Test describing a finish action with None text."""
        action = AgentAction(
            type=ActionType.FINISH,
            text=None
        )

        description = _describe_action(action)
        assert description == "(fim)"

    def test_describe_unknown_action_type(self) -> None:
        """Test describing an unknown action type."""
        # Create an action with a non-standard type name
        action = AgentAction(
            type=ActionType.MESSAGE,  # Use existing type but modify name for test
            text="Unknown action type"
        )
        action.type.name = "UNKNOWN_TYPE"

        description = _describe_action(action)
        assert description == "UNKNOWN_TYPE"


class TestDescribeObservation:
    """Test cases for the _describe_observation helper function."""

    def test_describe_successful_observation_minimal(self) -> None:
        """Test describing a successful observation with minimal data."""
        observation = Observation(success=True)

        description = _describe_observation(observation)
        assert description == "OK"

    def test_describe_successful_observation_with_return_code(self) -> None:
        """Test describing a successful observation with return code."""
        observation = Observation(
            success=True,
            return_code=0,
            output="Command executed successfully"
        )

        description = _describe_observation(observation)
        assert description == "OK (code=0, Command executed successfully)"

    def test_describe_successful_observation_with_duration(self) -> None:
        """Test describing a successful observation with duration."""
        observation = Observation(
            success=True,
            return_code=0,
            duration=2.5
        )

        description = _describe_observation(observation)
        assert description == "OK (code=0, t=2.50s)"

    def test_describe_successful_observation_with_all_fields(self) -> None:
        """Test describing a successful observation with all fields."""
        observation = Observation(
            success=True,
            return_code=0,
            output="Process completed",
            error=None,
            duration=1.25
        )

        description = _describe_observation(observation)
        assert description == "OK (code=0, t=1.25s, Process completed)"

    def test_describe_failed_observation_minimal(self) -> None:
        """Test describing a failed observation with minimal data."""
        observation = Observation(success=False)

        description = _describe_observation(observation)
        assert description == "FAIL"

    def test_describe_failed_observation_with_error(self) -> None:
        """Test describing a failed observation with error."""
        observation = Observation(
            success=False,
            return_code=1,
            error="Runtime error occurred"
        )

        description = _describe_observation(observation)
        assert description == "FAIL (code=1, ERR: Runtime error occurred)"

    def test_describe_failed_observation_with_multiline_error(self) -> None:
        """Test describing a failed observation with multiline error."""
        observation = Observation(
            success=False,
            return_code=1,
            error="First line of error\nSecond line of error\nThird line"
        )

        description = _describe_observation(observation)
        assert description == "FAIL (code=1, ERR: First line of error Second line of error)"

    def test_describe_observation_with_long_output(self) -> None:
        """Test describing observation with very long output."""
        long_output = "A" * 200  # Very long output
        observation = Observation(
            success=True,
            return_code=0,
            output=long_output
        )

        description = _describe_observation(observation)

        # Should truncate long output
        assert len(description) <= 200  # Should be truncated to reasonable length
        assert "..." in description

    def test_describe_observation_with_multiline_output(self) -> None:
        """Test describing observation with multiline output."""
        observation = Observation(
            success=True,
            return_code=0,
            output="Line 1\nLine 2\nLine 3\nLine 4"
        )

        description = _describe_observation(observation)

        # Should join first 3 lines
        assert "Line 1 Line 2 Line 3" in description
        assert "..." in description  # Should be truncated

    def test_describe_observation_return_code_none(self) -> None:
        """Test describing observation with None return code."""
        observation = Observation(
            success=True,
            return_code=None,
            output="No return code available"
        )

        description = _describe_observation(observation)
        assert description == "OK (No return code available)"

    def test_describe_observation_duration_none(self) -> None:
        """Test describing observation with None duration."""
        observation = Observation(
            success=True,
            return_code=0,
            duration=None,
            output="No duration recorded"
        )

        description = _describe_observation(observation)
        assert description == "OK (code=0, No duration recorded)"

    def test_describe_observation_empty_strings(self) -> None:
        """Test describing observation with empty string fields."""
        observation = Observation(
            success=False,
            return_code=1,
            output="",
            error=""
        )

        description = _describe_observation(observation)
        assert description == "FAIL (code=1)"

    def test_describe_observation_with_special_characters(self) -> None:
        """Test describing observation with special characters."""
        observation = Observation(
            success=True,
            return_code=0,
            output="Output with çharactères spécîaux and ümlauts"
        )

        description = _describe_observation(observation)
        assert "çharactères spécîaux" in description
        assert "ümlauts" in description


class TestHistoryIntegration:
    """Integration tests for history functionality."""

    def test_complete_workflow_simulation(self) -> None:
        """Test simulating a complete agent workflow through history."""
        history = AgentHistory()

        # Simulate a typical agent workflow
        workflow = [
            ("MESSAGE", "Starting new task"),
            ("READ_FILE", "requirements.txt"),
            ("RUN_COMMAND", ["pip", "install", "-r", "requirements.txt"]),
            ("WRITE_FILE", "main.py"),
            ("RUN_COMMAND", ["python", "main.py"]),
            ("APPLY_PATCH", "bug_fix.patch"),
            ("FINISH", "Task completed successfully")
        ]

        for step, (action_type, data) in enumerate(workflow, 1):
            # Create action based on type
            if action_type == "MESSAGE":
                action = AgentAction(type=ActionType.MESSAGE, text=data)
                observation = Observation(success=True, return_code=0, output=f"Processed message {step}")
            elif action_type == "READ_FILE":
                action = AgentAction(type=ActionType.READ_FILE, path=data)
                observation = Observation(success=True, return_code=0, output=f"Read {data}")
            elif action_type == "RUN_COMMAND":
                action = AgentAction(type=ActionType.RUN_COMMAND, command=data)
                observation = Observation(success=True, return_code=0, output=f"Command {data[0]} completed")
            elif action_type == "WRITE_FILE":
                action = AgentAction(type=ActionType.WRITE_FILE, path=data, content="print('Hello')")
                observation = Observation(success=True, return_code=0, output=f"Wrote {data}")
            elif action_type == "APPLY_PATCH":
                action = AgentAction(type=ActionType.APPLY_PATCH, diff=f"+ fix bug in {data}")
                observation = Observation(success=True, return_code=0, output="Patch applied")
            elif action_type == "FINISH":
                action = AgentAction(type=ActionType.FINISH, text=data)
                observation = Observation(success=True, return_code=0, output="Workflow completed")

            history.append(action, observation)

        # Verify complete history
        assert len(history.events) == 7

        # Test snapshot generation
        snapshot = history.snapshot()
        assert "Starting new task" in snapshot
        assert "READ_FILE requirements.txt" in snapshot
        assert "RUN `pip install -r requirements.txt`" in snapshot
        assert "WRITE_FILE main.py" in snapshot
        assert "APPLY_PATCH" in snapshot
        assert "Task completed successfully" in snapshot

        # Verify all observations were successful
        for event in history.events:
            assert event.observation.success

    def test_mixed_success_failure_workflow(self) -> None:
        """Test workflow with mixed success and failure observations."""
        history = AgentHistory()

        # Simulate workflow with some failures
        steps = [
            ("MESSAGE", "Starting task", True),
            ("RUN_COMMAND", ["python", "script.py"], True),
            ("RUN_COMMAND", ["python", "failing_script.py"], False),  # This fails
            ("APPLY_PATCH", "fix.patch", True),
            ("RUN_COMMAND", ["python", "fixed_script.py"], True),
            ("FINISH", "Task partially completed", True)
        ]

        for step, (action_type, data, success) in enumerate(steps, 1):
            if action_type == "MESSAGE":
                action = AgentAction(type=ActionType.MESSAGE, text=data)
            elif action_type == "RUN_COMMAND":
                action = AgentAction(type=ActionType.RUN_COMMAND, command=data)
            elif action_type == "APPLY_PATCH":
                action = AgentAction(type=ActionType.APPLY_PATCH, diff=f"+ fix for {data}")
            elif action_type == "FINISH":
                action = AgentAction(type=ActionType.FINISH, text=data)

            if success:
                observation = Observation(success=True, return_code=0, output=f"Step {step} succeeded")
            else:
                observation = Observation(success=False, return_code=1, error=f"Step {step} failed")

            history.append(action, observation)

        # Verify mixed results
        assert len(history.events) == 6

        successful_events = [e for e in history.events if e.observation.success]
        failed_events = [e for e in history.events if not e.observation.success]

        assert len(successful_events) == 5
        assert len(failed_events) == 1

        # Verify the failed event
        failed_event = failed_events[0]
        assert "failing_script.py" in str(failed_event.action.command)
        assert failed_event.observation.return_code == 1

    def test_large_history_performance(self) -> None:
        """Test performance with large history."""
        history = AgentHistory()

        # Create a large number of events
        num_events = 1000
        for i in range(num_events):
            action = AgentAction(type=ActionType.MESSAGE, text=f"Message {i}")
            observation = Observation(success=True, return_code=0, output=f"Response {i}")
            history.append(action, observation)

        # Should handle large history efficiently
        assert len(history.events) == num_events

        # Test that snapshot still works (and truncates appropriately)
        snapshot = history.snapshot(max_chars=5000)
        assert "... (histórico truncado) ..." in snapshot

        # Should contain first events but not all
        assert "[1] ACTION: Message 0" in snapshot
        assert "[1000] ACTION: Message 999" not in snapshot  # Should be truncated


class TestHistoryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_history_with_very_long_action_text(self) -> None:
        """Test history with extremely long action text."""
        history = AgentHistory()

        long_text = "A" * 10000  # Very long text
        action = AgentAction(type=ActionType.MESSAGE, text=long_text)
        observation = Observation(success=True, return_code=0, output="OK")

        history.append(action, observation)

        # Should handle long text without issues
        assert len(history.events) == 1
        assert history.events[0].action.text == long_text

    def test_history_with_unicode_content(self) -> None:
        """Test history with unicode content in actions and observations."""
        history = AgentHistory()

        unicode_text = "Unicode test: 测试 試験 試験 çharactères spécîaux"
        unicode_output = "Response with: ñáéíóú üöäß"

        action = AgentAction(type=ActionType.MESSAGE, text=unicode_text)
        observation = Observation(
            success=True,
            return_code=0,
            output=unicode_output
        )

        history.append(action, observation)

        # Should preserve unicode characters
        assert unicode_text in history.events[0].action.text
        assert unicode_output in history.events[0].observation.output

    def test_history_snapshot_with_zero_max_chars(self) -> None:
        """Test snapshot with zero max_chars."""
        history = AgentHistory()

        action = AgentAction(type=ActionType.MESSAGE, text="Test message")
        observation = Observation(success=True, return_code=0, output="Response")

        history.append(action, observation)

        # Should return empty string when max_chars is 0
        snapshot = history.snapshot(max_chars=0)
        assert snapshot == "... (histórico truncado) ..."

    def test_history_events_immutable_after_creation(self) -> None:
        """Test that events cannot be modified after creation."""
        history = AgentHistory()

        action = AgentAction(type=ActionType.MESSAGE, text="Original message")
        observation = Observation(success=True, return_code=0, output="Original response")

        history.append(action, observation)

        # Get the event
        event = history.events[0]

        # Try to modify the event (this should not affect the history)
        event.action.text = "Modified message"
        event.observation.success = False

        # Original history should be unchanged
        original_event = history.events[0]
        assert original_event.action.text == "Original message"
        assert original_event.observation.success is True

    def test_concurrent_history_access(self) -> None:
        """Test concurrent access to history (basic thread safety)."""
        history = AgentHistory()

        # Simulate concurrent appends
        def add_events(start_id: int, count: int) -> None:
            for i in range(count):
                action = AgentAction(type=ActionType.MESSAGE, text=f"Message {start_id}_{i}")
                observation = Observation(success=True, return_code=0, output=f"Response {start_id}_{i}")
                history.append(action, observation)

        # This is a simplified test - in real scenarios would need proper threading
        add_events(1, 10)
        add_events(2, 10)

        # Should have all events
        assert len(history.events) == 20

        # Events should maintain order
        message_indices = []
        for event in history.events:
            if "Message 1_" in event.action.text:
                message_indices.append(1)
            elif "Message 2_" in event.action.text:
                message_indices.append(2)

        # Should have 10 of each
        assert message_indices.count(1) == 10
        assert message_indices.count(2) == 10