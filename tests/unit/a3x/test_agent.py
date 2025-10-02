"""Testes para o módulo de agente A3X."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from a3x.actions import ActionType, AgentAction, Observation
from a3x.agent import AgentOrchestrator, AgentResult
from a3x.autoeval import AutoEvaluator
from a3x.config import GoalsConfig, LimitsConfig, TestSettings
from a3x.history import AgentHistory


class TestAgentResult:
    """Testes para a classe AgentResult."""

    def test_agent_result_creation(self) -> None:
        """Testa a criação de um resultado de agente."""
        history = AgentHistory()
        result = AgentResult(
            completed=True,
            iterations=10,
            failures=2,
            history=history,
            errors=["error1", "error2"]
        )

        assert result.completed is True
        assert result.iterations == 10
        assert result.failures == 2
        assert result.history == history
        assert result.errors == ["error1", "error2"]


class TestAgentOrchestrator:
    """Testes para a classe AgentOrchestrator."""

    def setup_method(self) -> None:
        """Configuração antes de cada teste."""
        # Criar mocks mais completos para simular a estrutura real do AgentConfig
        self.mock_config = Mock()
        self.mock_config.limits = Mock(spec=LimitsConfig)
        self.mock_config.limits.max_failures = 5
        self.mock_config.tests = Mock(spec=TestSettings)
        self.mock_config.tests.auto = False
        self.mock_config.tests.commands = []
        self.mock_config.goals = Mock(spec=GoalsConfig)
        self.mock_config.goals.get_threshold = Mock(return_value=0.8)
        # Adicionar os atributos necessários para a inicialização
        self.mock_config.workspace = Mock()
        self.mock_workspace_root = MagicMock(spec=Path)
        self.mock_config.workspace_root = self.mock_workspace_root
        self.mock_config.workspace.root = self.mock_workspace_root
        self.mock_config.audit = Mock()
        self.mock_config.audit.enable_file_log = True
        self.mock_config.audit.file_dir = "seed/changes"
        self.mock_config.audit.enable_git_commit = False
        self.mock_config.audit.commit_prefix = "A3X"
        self.mock_config.loop = Mock(use_memory=False)

        # Mock Path operations
        self.mock_hints_path = MagicMock(spec=Path)
        self.mock_hints_path.exists.return_value = False
        self.mock_hints_path.parent.mkdir.return_value = None
        self.mock_hints_path.with_suffix.return_value = self.mock_hints_path
        self.mock_hints_path.replace.return_value = None
        self.mock_workspace_root.__truediv__.return_value = self.mock_hints_path
        self.mock_workspace_root / "seed" / "policy_hints.json"  # Cache the path
        self.mock_workspace_root / "a3x" / "logs"  # For logging
        self.mock_workspace_root.mkdir.return_value = None
        self.mock_logs_path = self.mock_workspace_root / "a3x" / "logs" / "hints.log"
        self.mock_file_handler = Mock()
        self.mock_logger = Mock()
        self.mock_logger.addHandler.return_value = None

        self.mock_llm_client = Mock()
        # Configurar o mock do LLM client para retornar métricas adequadas
        self.mock_llm_client.get_last_metrics.return_value = {}
        self.mock_llm_client.notify_observation = Mock()
        self.mock_auto_evaluator = Mock(spec=AutoEvaluator)
        # Configurar o mock para retornar uma estrutura de dados válida para _read_metrics_history
        self.mock_auto_evaluator._read_metrics_history.return_value = {}
        # Configurar o mock para retornar um resumo vazio para latest_summary
        self.mock_auto_evaluator.latest_summary.return_value = ""

        # Criar um executor mock para evitar problemas na inicialização
        with patch("a3x.agent.ActionExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor

            self.orchestrator = AgentOrchestrator(
                config=self.mock_config,
                llm_client=self.mock_llm_client,
                auto_evaluator=self.mock_auto_evaluator
            )
            # Substituir o executor real pelo mock
            self.orchestrator.executor = mock_executor

    def test_initialization(self) -> None:
        """Testa a inicialização do orchestrator."""
        assert self.orchestrator.config == self.mock_config
        assert self.orchestrator.llm_client == self.mock_llm_client
        assert self.orchestrator.auto_evaluator == self.mock_auto_evaluator
        assert self.orchestrator._llm_metrics == {}
        assert self.orchestrator.recursion_depth == 3

    def test_initialization_without_auto_evaluator(self) -> None:
        """Testa a inicialização sem auto avaliador (deve criar um)."""
        # Mock para que AutoEvaluator funcione corretamente
        with patch("a3x.agent.AutoEvaluator") as mock_autoeval_class, \
             patch("a3x.agent.ActionExecutor") as mock_executor_class:
            mock_autoeval = Mock()
            mock_autoeval._read_metrics_history.return_value = {}
            mock_autoeval.latest_summary.return_value = ""
            mock_autoeval_class.return_value = mock_autoeval

            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor

            orchestrator = AgentOrchestrator(
                config=self.mock_config,
                llm_client=self.mock_llm_client
            )
            # Substituir o executor para evitar problemas nos outros testes
            orchestrator.executor = mock_executor

        assert orchestrator.auto_evaluator is not None

    @patch("a3x.agent.time.perf_counter")
    def test_run_method_success_with_finish_action(self, mock_time) -> None:
        """Testa o método run com sucesso ao receber ação FINISH."""
        mock_time.return_value = 100.0  # Fixed time for predictable duration

        # Configurar o histórico para não exceder limites
        self.mock_config.limits.max_iterations = 10

        # Configurar o LLM para retornar uma ação FINISH na primeira iteração
        finish_action = AgentAction(type=ActionType.FINISH, text="Goal achieved!")
        self.mock_llm_client.propose_action.return_value = finish_action

        observation = Observation(success=True, output="Finished successfully")
        self.mock_llm_client.notify_observation = Mock()

        # Mock the executor
        with patch("a3x.agent.ActionExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = observation
            mock_executor_class.return_value = mock_executor

            self.orchestrator.executor = mock_executor

            result = self.orchestrator.run("Test goal")

        assert result.completed is True
        assert result.iterations == 1
        assert result.failures == 0
        assert result.errors == []

        # Verify calls
        self.mock_llm_client.start.assert_called_once_with("Test goal")
        self.mock_llm_client.propose_action.assert_called_once()
        mock_executor.execute.assert_called_once_with(finish_action)
        self.mock_auto_evaluator.record.assert_called_once()

    @patch("a3x.agent.time.perf_counter")
    def test_run_method_max_iterations_reached(self, mock_time) -> None:
        """Testa o método run quando atinge o número máximo de iterações."""
        mock_time.return_value = 100.0  # Fixed time for predictable duration

        # O valor de recursion_depth é calculado dinamicamente com base em métricas anteriores
        # Com metrics_history vazio (mockado como {}), avg_success_rate será 0.0
        # O que resulta em recursion_depth = max(3, 3-1) = 3
        # Portanto, max_iterations será 10 * 3 = 30
        self.mock_config.limits.max_failures = 5

        # Configurar o LLM para retornar uma ação que não é FINISH
        continue_action = AgentAction(type=ActionType.MESSAGE, text="Continuing...")
        self.mock_llm_client.propose_action.return_value = continue_action

        observation = Observation(success=True, output="Continuing...")
        self.mock_llm_client.notify_observation = Mock()

        # Mock the executor
        with patch("a3x.agent.ActionExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = observation
            mock_executor_class.return_value = mock_executor

            self.orchestrator.executor = mock_executor

            result = self.orchestrator.run("Test goal")

        # Com recursion_depth inicial de 3 (calculado dinamicamente), teremos 10 * 3 = 30 iterações
        assert result.completed is False  # Não completou porque não recebeu FINISH
        assert result.iterations == 30  # Deveria ter feito 30 iterações (base_iterations * recursion_depth)
        assert len(result.errors) >= 1
        assert "Limite de iterações alcançado" in result.errors

    @patch("a3x.agent.time.perf_counter")
    def test_run_method_max_failures_exceeded(self, mock_time) -> None:
        """Testa o método run quando excede o número máximo de falhas."""
        mock_time.return_value = 100.0  # Fixed time for predictable duration

        # Configurar o número máximo de falhas
        self.mock_config.limits.max_failures = 2

        action = AgentAction(type=ActionType.MESSAGE, text="Action")
        self.mock_llm_client.propose_action.return_value = action

        # Criar uma observação falha
        failed_observation = Observation(success=False, output="Error occurred", error="Test error")

        # Mock the executor
        with patch("a3x.agent.ActionExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = failed_observation
            mock_executor_class.return_value = mock_executor

            self.orchestrator.executor = mock_executor

            result = self.orchestrator.run("Test goal")

        assert result.completed is False
        assert result.failures == 3  # 3 falhas porque o limite é 2 e então adiciona mais uma
        assert "Limite de falhas excedido" in result.errors

    def test_run_method_with_auto_test_enabled(self) -> None:
        """Testa o método run com testes automáticos habilitados."""
        # Configurar para ter testes automáticos
        self.mock_config.tests.auto = True
        self.mock_config.tests.commands = [["pytest", "tests/"]]
        self.mock_config.limits.max_iterations = 2

        # Configurar ações
        write_action = AgentAction(type=ActionType.WRITE_FILE, path="test.py", content="print('hello')")
        finish_action = AgentAction(type=ActionType.FINISH, text="Done")

        # Configurar LLM para retornar primeiro WRITE_FILE e depois FINISH
        call_count = 0
        def side_effect(state):
            nonlocal call_count
            call_count += 1
            return write_action if call_count == 1 else finish_action

        self.mock_llm_client.propose_action.side_effect = side_effect

        # Configurar observações
        write_obs = Observation(success=True, output="File written")
        finish_obs = Observation(success=True, output="Finished")
        test_obs = Observation(success=True, output="Tests passed")

        # Mock the executor
        with patch("a3x.agent.ActionExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.side_effect = [write_obs, test_obs, finish_obs]
            mock_executor_class.return_value = mock_executor

            self.orchestrator.executor = mock_executor
            self.orchestrator.llm_client.notify_observation = Mock()

            result = self.orchestrator.run("Test goal")

        assert result.completed is True
        # Verificar que execute foi chamado 3 vezes: write, test, finish
        assert mock_executor.execute.call_count >= 3

    def test_capture_llm_metrics(self) -> None:
        """Testa a captura de métricas do LLM."""
        # Configurar métricas do LLM
        self.mock_llm_client.get_last_metrics.return_value = {
            "tokens_used": 100,
            "response_time": 2.5,
            "temperature": 0.7
        }

        self.orchestrator._capture_llm_metrics()

        expected_metrics = {
            "tokens_used": [100.0],
            "response_time": [2.5],
            "temperature": [0.7]
        }
        assert self.orchestrator._llm_metrics == expected_metrics

        # Testar adicionando mais métricas
        self.mock_llm_client.get_last_metrics.return_value = {
            "tokens_used": 150,
            "response_time": 3.0
        }

        self.orchestrator._capture_llm_metrics()

        expected_metrics = {
            "tokens_used": [100.0, 150.0],
            "response_time": [2.5, 3.0],
            "temperature": [0.7]  # Não mudou na segunda chamada
        }
        assert self.orchestrator._llm_metrics == expected_metrics

    def test_capture_llm_metrics_with_non_numeric_values(self) -> None:
        """Testa a captura de métricas do LLM com valores não numéricos."""
        self.mock_llm_client.get_last_metrics.return_value = {
            "tokens_used": 100,
            "model_name": "gpt-4"  # Não numérico
        }

        self.orchestrator._capture_llm_metrics()

        # Apenas valores numéricos devem ser capturados
        expected_metrics = {
            "tokens_used": [100.0]
        }
        assert self.orchestrator._llm_metrics == expected_metrics

    def test_aggregate_llm_metrics(self) -> None:
        """Testa a agregação de métricas do LLM."""
        self.orchestrator._llm_metrics = {
            "tokens_used": [100, 200, 150],
            "response_time": [2.5, 3.0]
        }

        aggregated = self.orchestrator._aggregate_llm_metrics()

        expected = {
            "tokens_used_last": 150.0,
            "tokens_used_avg": 150.0,  # (100+200+150)/3
            "response_time_last": 3.0,
            "response_time_avg": 2.75  # (2.5+3.0)/2
        }

        for key, value in expected.items():
            assert aggregated[key] == pytest.approx(value)

    def test_aggregate_llm_metrics_empty(self) -> None:
        """Testa a agregação de métricas vazias."""
        self.orchestrator._llm_metrics = {}

        aggregated = self.orchestrator._aggregate_llm_metrics()

        assert aggregated == {}

    def test_notify_llm(self) -> None:
        """Testa a notificação do LLM sobre uma observação."""
        observation = Observation(success=True, output="A long output that might need truncation" * 100)

        self.orchestrator._notify_llm(observation)

        # Verificar que notify_observation foi chamado com o texto truncado
        self.mock_llm_client.notify_observation.assert_called_once()
        call_args = self.mock_llm_client.notify_observation.call_args[0][0]
        assert len(call_args) <= 2000  # Deve ser truncado
        assert call_args.endswith("...")  # Deve ter os três pontos finais

    def test_notify_llm_short_output(self) -> None:
        """Testa a notificação do LLM com saída curta."""
        observation = Observation(success=True, output="Short message")

        self.orchestrator._notify_llm(observation)

        # A saída curta não deve ser truncada
        self.mock_llm_client.notify_observation.assert_called_once_with("Short message")

    def test_run_auto_tests(self) -> None:
        """Testa a execução de testes automáticos."""
        # Configurar comandos de teste
        self.mock_config.tests.commands = [["pytest", "tests/unit"], ["ruff", "check", "."]]

        history_mock = Mock(spec=AgentHistory)
        self.mock_llm_client.notify_observation = Mock()

        test_obs1 = Observation(success=True, output="Tests passed")
        test_obs2 = Observation(success=True, output="Lint passed")

        with patch("a3x.agent.ActionExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.side_effect = [test_obs1, test_obs2]
            mock_executor_class.return_value = mock_executor

            self.orchestrator.executor = mock_executor

            self.orchestrator._run_auto_tests(history_mock)

        # Verificar que os testes foram executados
        assert mock_executor.execute.call_count == 2
        history_mock.append.assert_called()

    def test_analyze_history_basic_metrics(self) -> None:
        """Testa a análise de histórico para métricas básicas."""
        # Criar um histórico com algumas ações e observações
        history = AgentHistory()

        # Adicionar algumas ações de exemplo
        history.append(
            AgentAction(type=ActionType.WRITE_FILE, path="test.py", content="print('hello')"),
            Observation(success=True, output="File written")
        )
        history.append(
            AgentAction(type=ActionType.RUN_COMMAND, command=["echo", "test"]),
            Observation(success=True, output="test output")
        )
        history.append(
            AgentAction(type=ActionType.APPLY_PATCH, diff="some diff"),
            Observation(success=False, output="Patch failed", error="Error")
        )

        result_mock = Mock()
        result_mock.history = history
        result_mock.failures = 1
        result_mock.iterations = 3

        metrics, capabilities = self.orchestrator._analyze_history(result_mock)

        # Verificar as métricas esperadas
        assert "actions_total" in metrics
        assert "actions_success_rate" in metrics
        assert "apply_patch_count" in metrics
        assert "apply_patch_success_rate" in metrics
        assert "unique_commands" in metrics
        assert "unique_file_extensions" in metrics
        assert "failures" in metrics
        assert "iterations" in metrics

        # Verificar valores específicos
        assert metrics["actions_total"] == 3.0
        assert metrics["actions_success_rate"] == pytest.approx(2.0/3.0)  # 2 de 3 ações bem-sucedidas
        assert metrics["apply_patch_count"] == 1.0
        assert metrics["apply_patch_success_rate"] == pytest.approx(0.0)  # Patch falhou
        assert metrics["failures"] == 1.0
        assert metrics["iterations"] == 3.0

        # Verificar capacidades inferidas
        assert "core.diffing" in capabilities
        assert "horiz.python" in capabilities  # Por causa do arquivo .py


class TestAgentRefactoring:
    """Testes para métodos refatorados no AgentOrchestrator."""

    def setup_method(self) -> None:
        """Configuração para testes de refatoração."""
        self.mock_config = Mock()
        self.mock_config.workspace_root = MagicMock(spec=Path)
        self.mock_config.limits = Mock(spec=LimitsConfig)
        self.mock_config.tests = Mock(spec=TestSettings)
        self.mock_config.goals = Mock(spec=GoalsConfig)
        self.mock_config.goals.get_threshold = Mock(return_value=0.8)
        self.mock_config.audit = Mock()
        self.mock_config.audit.enable_file_log = True
        self.mock_config.audit.file_dir = "seed/changes"
        self.mock_config.audit.enable_git_commit = False
        self.mock_config.audit.commit_prefix = "A3X"
        self.mock_config.loop = Mock(use_memory=False)

        self.mock_llm_client = Mock()
        self.mock_llm_client.get_last_metrics.return_value = {}
        self.mock_llm_client.notify_observation = Mock()
        self.mock_llm_client.propose_action.return_value = AgentAction(type=ActionType.FINISH, text="Done")

        self.mock_auto_evaluator = Mock(spec=AutoEvaluator)
        self.mock_auto_evaluator._read_metrics_history.return_value = {}
        self.mock_auto_evaluator.latest_summary.return_value = ""

        with patch("a3x.agent.ActionExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor
            self.orchestrator = AgentOrchestrator(
                config=self.mock_config,
                llm_client=self.mock_llm_client,
                auto_evaluator=self.mock_auto_evaluator
            )
            self.orchestrator.executor = mock_executor

    @patch("a3x.agent.PlanComposer")
    def test_decompose_goal(self, mock_composer):
        """Testa _decompose_goal para goals complexos."""
        mock_composer_instance = Mock()
        mock_composer.return_value = mock_composer_instance
        mock_composer_instance.decompose_goal.return_value = ["subtask1", "subtask2"]

        goal = "Complex goal: implement feature A and B"
        subtasks = self.orchestrator._decompose_goal(goal)

        assert len(subtasks) == 2
        assert "subtask1" in subtasks
        mock_composer_instance.decompose_goal.assert_called_once_with(goal)

        # Verify subgoals saved
        subgoals_path = self.mock_config.workspace_root / "seed/subgoals.json"
        mock_open = Mock()
        with patch("builtins.open", mock_open):
            self.orchestrator._decompose_goal(goal)
            mock_open.assert_called()

    @patch("a3x.agent.PlanComposer")
    def test_handle_subtasks(self, mock_composer):
        """Testa _handle_subtasks para múltiplas subtasks."""
        mock_composer_instance = Mock()
        mock_composer.return_value = mock_composer_instance
        mock_composer_instance.execute_plan.return_value = [{"success": True}, {"success": False}]

        subtasks = ["task1", "task2"]
        history = AgentHistory()
        goal = "parent goal"

        result = self.orchestrator._handle_subtasks(subtasks, history, goal)

        assert result.completed is True
        assert result.iterations == 1
        assert result.failures == 0
        assert len(history.events) == 1
        mock_composer_instance.execute_plan.assert_called_once_with(subtasks, parent_agent=self.orchestrator)

    def test_constants_usage(self):
        """Testa uso de constantes como DEFAULT_MAX_SUB_DEPTH."""
        assert self.orchestrator.DEFAULT_MAX_SUB_DEPTH == 3
        assert self.orchestrator.max_sub_depth == 3  # Default from hints

        # Test adjustment
        self.orchestrator.hints = {"max_sub_depth": 5}
        self.orchestrator.max_sub_depth = int(self.orchestrator.hints.get("max_sub_depth", self.orchestrator.DEFAULT_MAX_SUB_DEPTH))
        assert self.orchestrator.max_sub_depth == 5

    @pytest.mark.parametrize("goal_complexity, expected_subtasks", [
        ("simple task", 1),
        ("implement A and B", 2),
        ("plan and execute multi-step", 3),
    ])
    @patch("a3x.agent.PlanComposer")
    def test_run_with_subtasks_parametrized(self, mock_composer, goal_complexity, expected_subtasks):
        """Testa run variants com subtasks parametrizadas."""
        mock_composer_instance = Mock()
        mock_composer.return_value = mock_composer_instance
        mock_composer_instance.decompose_goal.return_value = [f"sub_{i}" for i in range(expected_subtasks)]

        goal = f"{goal_complexity} goal"
        result = self.orchestrator.run(goal)

        if expected_subtasks > 1:
            assert result.iterations == 1  # Handled by _handle_subtasks
            mock_composer_instance.execute_plan.assert_called()
        else:
            assert result.completed is True
            mock_composer_instance.decompose_goal.assert_called()

    def test_run_edge_case_max_depth(self):
        """Testa edge case: run com depth max, no decomposition."""
        self.orchestrator.depth = self.orchestrator.max_sub_depth
        goal = "deep nested goal"
        result = self.orchestrator.run(goal)

        assert result.completed is True

    @pytest.mark.parametrize("success_rate, expected_adjust", [
        (0.9, 1),  # High, increase
        (0.5, -1),  # Low, decrease
        (0.7, 0),   # Medium, no change
    ])
    def test_adjust_recursion_depth_parametrized(self, success_rate, expected_adjust):
        """Testa _adjust_recursion_depth parametrizado."""
        self.mock_auto_evaluator._read_metrics_history.return_value = {"actions_success_rate": [success_rate] * 3}
        initial_depth = self.orchestrator.recursion_depth

        self.orchestrator._adjust_recursion_depth()

        adjusted = self.orchestrator.recursion_depth
        delta = adjusted - initial_depth
        assert delta == expected_adjust

    def test_gather_memory_lessons_no_memory(self):
        """Testa _gather_memory_lessons quando memory desabilitado."""
        self.mock_config.loop.use_memory = False
        lessons = self.orchestrator._gather_memory_lessons("goal")
        assert lessons == ""

    @patch("a3x.agent.SemanticMemory")
    def test_gather_memory_lessons_with_memory(self, mock_memory):
        """Testa _gather_memory_lessons com memória semântica."""
        self.mock_config.loop.use_memory = True
        self.mock_config.loop.memory_top_k = 2

        mock_mem_instance = Mock()
        mock_memory.return_value = mock_mem_instance
        mock_mem_instance.query.return_value = [("lesson1", 0.9), ("lesson2", 0.8)]

        lessons = self.orchestrator._gather_memory_lessons("test goal")

        assert "Useful lessons:" in lessons
        assert "1. lesson1" in lessons
        assert "2. lesson2" in lessons
        mock_mem_instance.query.assert_called_once_with("test goal", top_k=2)

class TestAgentRefactoring:
    """Testes para métodos refatorados no AgentOrchestrator."""

    def setup_method(self) -> None:
        """Configuração para testes de refatoração."""
        self.mock_config = Mock()
        self.mock_config.workspace_root = Path("/tmp/test_workspace")
        self.mock_config.limits = Mock(spec=LimitsConfig)
        self.mock_config.tests = Mock(spec=TestSettings)
        self.mock_config.goals = Mock(spec=GoalsConfig)
        self.mock_config.goals.get_threshold = Mock(return_value=0.8)
        self.mock_config.audit = Mock()
        self.mock_config.audit.enable_file_log = True
        self.mock_config.audit.file_dir = "seed/changes"
        self.mock_config.audit.enable_git_commit = False
        self.mock_config.audit.commit_prefix = "A3X"
        self.mock_config.loop = Mock(use_memory=False)

        self.mock_llm_client = Mock()
        self.mock_llm_client.get_last_metrics.return_value = {}
        self.mock_llm_client.notify_observation = Mock()
        self.mock_llm_client.propose_action.return_value = AgentAction(type=ActionType.FINISH, text="Done")

        self.mock_auto_evaluator = Mock(spec=AutoEvaluator)
        self.mock_auto_evaluator._read_metrics_history.return_value = {}
        self.mock_auto_evaluator.latest_summary.return_value = ""

        with patch("a3x.agent.ActionExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor
            self.orchestrator = AgentOrchestrator(
                config=self.mock_config,
                llm_client=self.mock_llm_client,
                auto_evaluator=self.mock_auto_evaluator
            )
            self.orchestrator.executor = mock_executor

    @patch("a3x.agent.PlanComposer")
    def test_decompose_goal(self, mock_composer):
        """Testa _decompose_goal para goals complexos."""
        mock_composer_instance = Mock()
        mock_composer.return_value = mock_composer_instance
        mock_composer_instance.decompose_goal.return_value = ["subtask1", "subtask2"]

        goal = "Complex goal: implement feature A and B"
        subtasks = self.orchestrator._decompose_goal(goal)

        assert len(subtasks) == 2
        assert "subtask1" in subtasks
        mock_composer_instance.decompose_goal.assert_called_once_with(goal)

        # Verify subgoals saved
        subgoals_path = self.mock_config.workspace_root / "seed/subgoals.json"
        mock_open = Mock()
        with patch("builtins.open", mock_open):
            self.orchestrator._decompose_goal(goal)
            mock_open.assert_called()

    @patch("a3x.agent.PlanComposer")
    def test_handle_subtasks(self, mock_composer):
        """Testa _handle_subtasks para múltiplas subtasks."""
        mock_composer_instance = Mock()
        mock_composer.return_value = mock_composer_instance
        mock_composer_instance.execute_plan.return_value = [{"success": True}, {"success": False}]

        subtasks = ["task1", "task2"]
        history = AgentHistory()
        goal = "parent goal"

        result = self.orchestrator._handle_subtasks(subtasks, history, goal)

        assert result.completed is True
        assert result.iterations == 1
        assert result.failures == 0
        assert len(history.events) == 1
        mock_composer_instance.execute_plan.assert_called_once_with(subtasks, parent_agent=self.orchestrator)

    @patch("a3x.agent.PlanComposer")
    def test_handle_subtasks_max_depth(self, mock_composer):
        """Testa _handle_subtasks quando max_sub_depth é atingido."""
        self.orchestrator.depth = self.orchestrator.max_sub_depth
        subtasks = self.orchestrator._decompose_goal("complex goal")  # Would be multiple, but depth max
        assert len(subtasks) == 1  # Treated as single task

        history = AgentHistory()
        goal = "parent goal"
        result = self.orchestrator._handle_subtasks(subtasks, history, goal)

        assert result.completed is True
        # Note: logger.info is mocked, but we can check if max depth logic applied

    def test_constants_usage(self):
        """Testa uso de constantes como DEFAULT_MAX_SUB_DEPTH."""
        assert self.orchestrator.DEFAULT_MAX_SUB_DEPTH == 3
        assert self.orchestrator.max_sub_depth == 3  # Default from hints

        # Test adjustment
        self.orchestrator.hints = {"max_sub_depth": 5}
        self.orchestrator.max_sub_depth = int(self.orchestrator.hints.get("max_sub_depth", self.orchestrator.DEFAULT_MAX_SUB_DEPTH))
        assert self.orchestrator.max_sub_depth == 5

    @pytest.mark.parametrize("goal_complexity, expected_subtasks", [
        ("simple task", 1),
        ("implement A and B", 2),
        ("plan and execute multi-step", 3),
    ])
    @patch("a3x.agent.PlanComposer")
    def test_run_with_subtasks_parametrized(self, mock_composer, goal_complexity, expected_subtasks):
        """Testa run variants com subtasks parametrizadas."""
        mock_composer_instance = Mock()
        mock_composer.return_value = mock_composer_instance
        mock_composer_instance.decompose_goal.return_value = [f"sub_{i}" for i in range(expected_subtasks)]

        goal = f"{goal_complexity} goal"
        result = self.orchestrator.run(goal)

        if expected_subtasks > 1:
            assert result.iterations == 1  # Handled by _handle_subtasks
            mock_composer_instance.execute_plan.assert_called()
        else:
            assert result.completed is True
            mock_composer_instance.decompose_goal.assert_called()

    def test_run_edge_case_max_depth(self):
        """Testa edge case: run com depth max, no decomposition."""
        self.orchestrator.depth = self.orchestrator.max_sub_depth
        goal = "deep nested goal"
        result = self.orchestrator.run(goal)

        assert result.completed is True

    @pytest.mark.parametrize("success_rate, expected_adjust", [
        (0.9, 1),  # High, increase
        (0.5, -1),  # Low, decrease
        (0.7, 0),   # Medium, no change
    ])
    def test_adjust_recursion_depth_parametrized(self, success_rate, expected_adjust):
        """Testa _adjust_recursion_depth parametrizado."""
        self.mock_auto_evaluator._read_metrics_history.return_value = {"actions_success_rate": [success_rate] * 3}
        initial_depth = self.orchestrator.recursion_depth

        self.orchestrator._adjust_recursion_depth()

        adjusted = self.orchestrator.recursion_depth
        delta = adjusted - initial_depth
        assert delta == expected_adjust

    def test_gather_memory_lessons_no_memory(self):
        """Testa _gather_memory_lessons quando memory desabilitado."""
        self.mock_config.loop.use_memory = False
        lessons = self.orchestrator._gather_memory_lessons("goal")
        assert lessons == ""

    @patch("a3x.agent.SemanticMemory")
    def test_gather_memory_lessons_with_memory(self, mock_memory):
        """Testa _gather_memory_lessons com memória semântica."""
        self.mock_config.loop.use_memory = True
        self.mock_config.loop.memory_top_k = 2

        mock_mem_instance = Mock()
        mock_memory.return_value = mock_mem_instance
        mock_mem_instance.query.return_value = [("lesson1", 0.9), ("lesson2", 0.8)]

        lessons = self.orchestrator._gather_memory_lessons("test goal")

        assert "Useful lessons:" in lessons
        assert "1. lesson1" in lessons
        assert "2. lesson2" in lessons
        mock_mem_instance.query.assert_called_once_with("test goal", top_k=2)



class TestAgentAutonomousModeStatus(TestAgentRefactoring):
    """Testes específicos para o status do modo autônomo."""

    def setup_method(self) -> None:
        self._seed_strategist_patcher = patch("a3x.agent.LLMSeedStrategist")
        self.mock_seed_strategist = self._seed_strategist_patcher.start()
        self._meta_recursion_patcher = patch("a3x.agent.MetaRecursionEngine")
        self.mock_meta_recursion = self._meta_recursion_patcher.start()
        super().setup_method()

    def teardown_method(self) -> None:
        self._meta_recursion_patcher.stop()
        self._seed_strategist_patcher.stop()

    def test_get_autonomous_mode_status_uses_config_enable(self):
        """Garante que o status do modo autônomo usa o flag correto."""
        self.orchestrator._autonomous_config.enable = False

        status = self.orchestrator.get_autonomous_mode_status()

        assert status["autonomous_mode_enabled"] is False
        assert isinstance(status["configuration"], dict)
        assert status["configuration"].get("enable") is False
