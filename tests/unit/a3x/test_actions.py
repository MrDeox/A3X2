"""Testes para o módulo de ações do agente A3X."""

import pytest
from a3x.actions import ActionType, AgentAction, Observation, AgentState


class TestActionType:
    """Testes para os tipos de ações."""
    
    def test_action_types_exist(self) -> None:
        """Verifica que todos os tipos de ação existem."""
        assert hasattr(ActionType, 'MESSAGE')
        assert hasattr(ActionType, 'RUN_COMMAND')
        assert hasattr(ActionType, 'APPLY_PATCH')
        assert hasattr(ActionType, 'WRITE_FILE')
        assert hasattr(ActionType, 'READ_FILE')
        assert hasattr(ActionType, 'SELF_MODIFY')
        assert hasattr(ActionType, 'FINISH')


class TestAgentAction:
    """Testes para a classe AgentAction."""
    
    def test_agent_action_creation(self) -> None:
        """Testa a criação de uma ação do agente."""
        action = AgentAction(
            type=ActionType.RUN_COMMAND,
            command=["echo", "hello"],
            cwd="/tmp"
        )
        assert action.type == ActionType.RUN_COMMAND
        assert action.command == ["echo", "hello"]
        assert action.cwd == "/tmp"
        assert action.dry_run is False

    def test_agent_action_defaults(self) -> None:
        """Testa os valores padrão de AgentAction."""
        action = AgentAction(type=ActionType.MESSAGE)
        assert action.text is None
        assert action.command is None
        assert action.cwd is None
        assert action.diff is None
        assert action.path is None
        assert action.content is None
        assert action.dry_run is False
        assert action.metadata == {}

    def test_agent_action_with_all_fields(self) -> None:
        """Testa AgentAction com todos os campos preenchidos."""
        metadata = {"test": "value"}
        action = AgentAction(
            type=ActionType.WRITE_FILE,
            text="test message",
            command=["python", "script.py"],
            cwd="/home/user",
            diff="diff content",
            path="/tmp/file.py",
            content="file content",
            dry_run=True,
            metadata=metadata
        )
        assert action.type == ActionType.WRITE_FILE
        assert action.text == "test message"
        assert action.command == ["python", "script.py"]
        assert action.cwd == "/home/user"
        assert action.diff == "diff content"
        assert action.path == "/tmp/file.py"
        assert action.content == "file content"
        assert action.dry_run is True
        assert action.metadata == metadata


class TestObservation:
    """Testes para a classe Observation."""
    
    def test_observation_creation(self) -> None:
        """Testa a criação de uma observação."""
        observation = Observation(
            success=True,
            output="test output",
            error="test error",
            return_code=0,
            duration=1.5,
            type="test_type",
            metadata={"key": "value"}
        )
        assert observation.success is True
        assert observation.output == "test output"
        assert observation.error == "test error"
        assert observation.return_code == 0
        assert observation.duration == 1.5
        assert observation.type == "test_type"
        assert observation.metadata == {"key": "value"}

    def test_observation_defaults(self) -> None:
        """Testa os valores padrão de Observation."""
        observation = Observation(success=False)
        assert observation.success is False
        assert observation.output == ""
        assert observation.error is None
        assert observation.return_code is None
        assert observation.duration == 0.0
        assert observation.type == "generic"
        assert observation.metadata == {}

    def test_successful_observation(self) -> None:
        """Testa uma observação de sucesso."""
        observation = Observation(success=True, output="Success!")
        assert observation.success is True
        assert observation.output == "Success!"
        assert observation.error is None

    def test_failed_observation(self) -> None:
        """Testa uma observação de falha."""
        observation = Observation(success=False, error="Something went wrong")
        assert observation.success is False
        assert observation.error == "Something went wrong"


class TestAgentState:
    """Testes para a classe AgentState."""
    
    def test_agent_state_creation(self) -> None:
        """Testa a criação de um estado do agente."""
        state = AgentState(
            goal="test goal",
            history_snapshot="test history",
            iteration=5,
            max_iterations=10,
            seed_context="test context"
        )
        assert state.goal == "test goal"
        assert state.history_snapshot == "test history"
        assert state.iteration == 5
        assert state.max_iterations == 10
        assert state.seed_context == "test context"

    def test_agent_state_defaults(self) -> None:
        """Testa os valores padrão de AgentState."""
        state = AgentState(
            goal="minimal goal",
            history_snapshot="snapshot",
            iteration=1,
            max_iterations=10
        )
        assert state.seed_context == ""