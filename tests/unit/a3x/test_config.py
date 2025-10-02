"""Testes para o módulo de configuração do agente A3X."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from a3x.config import (
    AgentConfig,
    AuditConfig,
    GoalsConfig,
    LimitsConfig,
    LLMConfig,
    LoopConfig,
    PoliciesConfig,
    TestSettings,
    WorkspaceConfig,
    _read_yaml,
    load_config,
)


class TestLLMConfig:
    """Testes para a configuração do LLM."""

    def test_llm_config_creation(self) -> None:
        """Testa a criação da configuração do LLM."""
        llm_config = LLMConfig(
            type="openai",
            model="gpt-4",
            api_key_env="OPENAI_API_KEY",
            endpoint="https://api.openai.com/v1",
            base_url="https://api.openai.com"
        )
        assert llm_config.type == "openai"
        assert llm_config.model == "gpt-4"
        assert llm_config.api_key_env == "OPENAI_API_KEY"
        assert llm_config.endpoint == "https://api.openai.com/v1"
        assert llm_config.base_url == "https://api.openai.com"


class TestWorkspaceConfig:
    """Testes para a configuração do workspace."""

    def test_workspace_config_creation(self) -> None:
        """Testa a criação da configuração do workspace."""
        workspace_config = WorkspaceConfig(
            root=Path("/home/user/project"),
            allow_outside_root=True
        )
        assert workspace_config.root == Path("/home/user/project")
        assert workspace_config.allow_outside_root is True

    def test_workspace_config_defaults(self) -> None:
        """Testa os valores padrão da configuração do workspace."""
        workspace_config = WorkspaceConfig()
        assert workspace_config.root == Path()
        assert workspace_config.allow_outside_root is False


class TestLimitsConfig:
    """Testes para a configuração de limites."""

    def test_limits_config_creation(self) -> None:
        """Testa a criação da configuração de limites."""
        limits_config = LimitsConfig(
            max_iterations=100,
            command_timeout=60,
            max_failures=5,
            total_timeout=3600
        )
        assert limits_config.max_iterations == 100
        assert limits_config.command_timeout == 60
        assert limits_config.max_failures == 5
        assert limits_config.total_timeout == 3600

    def test_limits_config_defaults(self) -> None:
        """Testa os valores padrão da configuração de limites."""
        limits_config = LimitsConfig()
        assert limits_config.max_iterations == 50
        assert limits_config.command_timeout == 120
        assert limits_config.max_failures == 10
        assert limits_config.total_timeout is None


class TestTestSettings:
    """Testes para as configurações de testes."""

    def test_test_settings_creation(self) -> None:
        """Testa a criação das configurações de testes."""
        test_commands = [["pytest", "tests/"], ["ruff", "check", "."]]
        test_settings = TestSettings(
            auto=True,
            commands=test_commands
        )
        assert test_settings.auto is True
        assert test_settings.commands == test_commands

    def test_test_settings_defaults(self) -> None:
        """Testa os valores padrão das configurações de testes."""
        test_settings = TestSettings()
        assert test_settings.auto is False
        assert test_settings.commands == []


class TestPoliciesConfig:
    """Testes para as configurações de políticas."""

    def test_policies_config_creation(self) -> None:
        """Testa a criação das configurações de políticas."""
        deny_commands = ["rm -rf", "sudo"]
        policies_config = PoliciesConfig(
            allow_network=True,
            allow_shell_write=False,
            deny_commands=deny_commands
        )
        assert policies_config.allow_network is True
        assert policies_config.allow_shell_write is False
        assert policies_config.deny_commands == deny_commands

    def test_policies_config_defaults(self) -> None:
        """Testa os valores padrão das configurações de políticas."""
        policies_config = PoliciesConfig()
        assert policies_config.allow_network is False
        assert policies_config.allow_shell_write is True
        assert policies_config.deny_commands == []


class TestGoalsConfig:
    """Testes para as configurações de metas."""

    def test_goals_config_creation(self) -> None:
        """Testa a criação das configurações de metas."""
        thresholds = {
            "apply_patch_success_rate": 0.85,
            "actions_success_rate": 0.90,
            "tests_success_rate": 0.95
        }
        goals_config = GoalsConfig(thresholds=thresholds)
        assert goals_config.thresholds == thresholds

    def test_goals_config_get_threshold(self) -> None:
        """Testa o método de obtenção de threshold."""
        goals_config = GoalsConfig(thresholds={"metric1": 0.8})
        assert goals_config.get_threshold("metric1", 0.5) == pytest.approx(0.8)
        assert goals_config.get_threshold("nonexistent", 0.5) == pytest.approx(0.5)

    def test_goals_config_defaults(self) -> None:
        """Testa os valores padrão das configurações de metas."""
        goals_config = GoalsConfig()
        assert goals_config.thresholds == {}


class TestLoopConfig:
    """Testes para as configurações de loop."""

    def test_loop_config_creation(self) -> None:
        """Testa a criação das configurações de loop."""
        loop_config = LoopConfig(
            auto_seed=True,
            seed_backlog=Path("seed/backlog.yaml"),
            seed_config=Path("configs/seed.yaml"),
            seed_interval=5.0,
            seed_max_runs=10,
            stop_when_idle=False
        )
        assert loop_config.auto_seed is True
        assert loop_config.seed_backlog == Path("seed/backlog.yaml")
        assert loop_config.seed_config == Path("configs/seed.yaml")
        assert loop_config.seed_interval == 5.0
        assert loop_config.seed_max_runs == 10
        assert loop_config.stop_when_idle is False

    def test_loop_config_defaults(self) -> None:
        """Testa os valores padrão das configurações de loop."""
        loop_config = LoopConfig()
        assert loop_config.auto_seed is False
        assert loop_config.seed_backlog == Path("seed/backlog.yaml")
        assert loop_config.seed_config is None
        assert loop_config.seed_interval == 0.0
        assert loop_config.seed_max_runs is None
        assert loop_config.stop_when_idle is True


class TestAuditConfig:
    """Testes para as configurações de auditoria."""

    def test_audit_config_creation(self) -> None:
        """Testa a criação das configurações de auditoria."""
        audit_config = AuditConfig(
            enable_file_log=False,
            file_dir=Path("changes"),
            enable_git_commit=True,
            commit_prefix="TEST"
        )
        assert audit_config.enable_file_log is False
        assert audit_config.file_dir == Path("changes")
        assert audit_config.enable_git_commit is True
        assert audit_config.commit_prefix == "TEST"

    def test_audit_config_defaults(self) -> None:
        """Testa os valores padrão das configurações de auditoria."""
        audit_config = AuditConfig()
        assert audit_config.enable_file_log is True
        assert audit_config.file_dir == Path("seed/changes")
        assert audit_config.enable_git_commit is False
        assert audit_config.commit_prefix == "A3X"


class TestAgentConfig:
    """Testes para a configuração completa do agente."""

    def test_agent_config_creation(self) -> None:
        """Testa a criação da configuração do agente."""
        llm_config = LLMConfig(type="openai")
        workspace_config = WorkspaceConfig()
        limits_config = LimitsConfig()
        test_settings = TestSettings()
        policies_config = PoliciesConfig()
        goals_config = GoalsConfig()
        loop_config = LoopConfig()
        audit_config = AuditConfig()

        agent_config = AgentConfig(
            llm=llm_config,
            workspace=workspace_config,
            limits=limits_config,
            tests=test_settings,
            policies=policies_config,
            goals=goals_config,
            loop=loop_config,
            audit=audit_config
        )

        assert agent_config.llm == llm_config
        assert agent_config.workspace == workspace_config
        assert agent_config.limits == limits_config
        assert agent_config.tests == test_settings
        assert agent_config.policies == policies_config
        assert agent_config.goals == goals_config
        assert agent_config.loop == loop_config
        assert agent_config.audit == audit_config

    def test_agent_config_workspace_root_property(self) -> None:
        """Testa a propriedade workspace_root."""
        workspace_config = WorkspaceConfig(root=Path("/tmp/test"))
        agent_config = AgentConfig(
            llm=LLMConfig(type="openai"),
            workspace=workspace_config,
            limits=LimitsConfig(),
            tests=TestSettings(),
            policies=PoliciesConfig(),
            goals=GoalsConfig(),
            loop=LoopConfig(),
            audit=AuditConfig()
        )

        expected = Path("/tmp/test").resolve()
        assert agent_config.workspace_root == expected


class TestReadYaml:
    """Testes para a função _read_yaml."""

    @patch("a3x.config.Path.open", new_callable=mock_open, read_data="key: value")
    @patch("a3x.config.Path.exists", return_value=True)
    def test_read_yaml_success(self, mock_exists, mock_file) -> None:
        """Testa a leitura bem-sucedida de um arquivo YAML."""
        result = _read_yaml(Path("test.yaml"))

        assert result == {"key": "value"}
        mock_file.assert_called_once_with("r", encoding="utf-8")

    @patch("a3x.config.Path.exists", return_value=False)
    def test_read_yaml_file_not_found(self, mock_exists) -> None:
        """Testa o erro quando o arquivo não é encontrado."""
        with pytest.raises(FileNotFoundError):
            _read_yaml(Path("nonexistent.yaml"))

    @patch("a3x.config.Path.open", new_callable=mock_open, read_data="")
    @patch("a3x.config.Path.exists", return_value=True)
    def test_read_yaml_empty_file(self, mock_exists, mock_file) -> None:
        """Testa a leitura de um arquivo YAML vazio."""
        result = _read_yaml(Path("empty.yaml"))

        assert result == {}

    @patch("a3x.config.Path.open", new_callable=mock_open, read_data="invalid: [unclosed list")
    @patch("a3x.config.Path.exists", return_value=True)
    def test_read_yaml_invalid_content(self, mock_exists, mock_file) -> None:
        """Testa o erro quando o conteúdo do YAML é inválido."""
        with patch("a3x.config.yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
            with pytest.raises(yaml.YAMLError):
                _read_yaml(Path("invalid.yaml"))

    @patch("a3x.config.Path.open", new_callable=mock_open, read_data="not a dict")
    @patch("a3x.config.Path.exists", return_value=True)
    def test_read_yaml_non_dict_root(self, mock_exists, mock_file) -> None:
        """Testa o erro quando o conteúdo do YAML não é um dicionário."""
        with patch("a3x.config.yaml.safe_load", return_value="not a dict"):
            with pytest.raises(ValueError):
                _read_yaml(Path("non_dict.yaml"))


class TestLoadConfig:
    """Testes para a função load_config."""

    @patch("a3x.config._read_yaml")
    @patch("pathlib.Path.exists")
    def test_load_config_minimal(self, mock_exists, mock_read_yaml) -> None:
        """Testa o carregamento de uma configuração mínima."""
        mock_exists.return_value = True
        mock_read_yaml.return_value = {
            "llm": {"type": "openai"}
        }

        config = load_config(Path("test.yaml"))

        assert config.llm.type == "openai"
        assert config.workspace.root == Path().resolve()
        assert config.limits.max_iterations == 50

    @patch("a3x.config._read_yaml")
    @patch("pathlib.Path.exists")
    def test_load_config_full(self, mock_exists, mock_read_yaml) -> None:
        """Testa o carregamento de uma configuração completa."""
        mock_exists.return_value = True
        mock_read_yaml.return_value = {
            "llm": {
                "type": "openrouter",
                "model": "test-model",
                "api_key_env": "TEST_API_KEY",
                "endpoint": "https://test.api/v1",
                "base_url": "https://test.api"
            },
            "workspace": {
                "root": "/tmp/workspace",
                "allow_outside_root": True
            },
            "limits": {
                "max_iterations": 100,
                "command_timeout": 300,
                "max_failures": 3,
                "total_timeout": 7200
            },
            "tests": {
                "auto": True,
                "commands": [
                    ["pytest", "tests/"],
                    ["ruff", "check", "."]
                ]
            },
            "policies": {
                "allow_network": True,
                "allow_shell_write": False,
                "deny_commands": ["rm -rf", "sudo"]
            },
            "goals": {
                "apply_patch_success_rate": 0.9,
                "actions_success_rate": {"min": 0.85}
            },
            "loop": {
                "auto_seed": True,
                "seed_backlog": "custom/backlog.yaml",
                "seed_config": "custom/seed_config.yaml",
                "seed_interval": 10.0,
                "seed_max_runs": 5,
                "stop_when_idle": False
            },
            "audit": {
                "enable_file_log": False,
                "file_dir": "custom/changes",
                "enable_git_commit": True,
                "commit_prefix": "CUSTOM"
            }
        }

        config = load_config(Path("test.yaml"))

        # Test LLM config
        assert config.llm.type == "openrouter"
        assert config.llm.model == "test-model"
        assert config.llm.api_key_env == "TEST_API_KEY"
        assert config.llm.endpoint == "https://test.api/v1"
        assert config.llm.base_url == "https://test.api"

        # Test workspace config
        assert config.workspace.root == Path("/tmp/workspace").resolve()
        assert config.workspace.allow_outside_root is True

        # Test limits config
        assert config.limits.max_iterations == 100
        assert config.limits.command_timeout == 300
        assert config.limits.max_failures == 3
        assert config.limits.total_timeout == 7200

        # Test tests config
        assert config.tests.auto is True
        assert config.tests.commands == [["pytest", "tests/"], ["ruff", "check", "."]]

        # Test policies config
        assert config.policies.allow_network is True
        assert config.policies.allow_shell_write is False
        assert config.policies.deny_commands == ["rm -rf", "sudo"]

        # Test goals config
        assert config.goals.thresholds["apply_patch_success_rate"] == 0.9
        assert config.goals.thresholds["actions_success_rate"] == 0.85

        # Test loop config
        assert config.loop.auto_seed is True
        assert config.loop.seed_backlog == Path("custom/backlog.yaml").resolve()
        assert config.loop.seed_config == Path("custom/seed_config.yaml").resolve()
        assert config.loop.seed_interval == 10.0
        assert config.loop.seed_max_runs == 5
        assert config.loop.stop_when_idle is False

        # Test audit config
        assert config.audit.enable_file_log is False
        assert config.audit.file_dir == Path("custom/changes")
        assert config.audit.enable_git_commit is True
        assert config.audit.commit_prefix == "CUSTOM"
