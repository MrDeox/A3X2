"""Carregamento e validação de configuração."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

try:
    from a3x.config.validation import validate_config_file, ValidationError
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False


@dataclass
class LLMConfig:
    type: str
    model: str | None = None
    script: Path | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None


@dataclass
class WorkspaceConfig:
    root: Path = Path()
    allow_outside_root: bool = False


@dataclass
class LimitsConfig:
    max_iterations: int = 50
    command_timeout: int = 120
    max_failures: int = 10
    total_timeout: int | None = None


@dataclass
class TestSettings:
    auto: bool = False
    commands: list[list[str]] = field(default_factory=list)


@dataclass
class PoliciesConfig:
    allow_network: bool = False
    allow_shell_write: bool = True
    deny_commands: list[str] = field(default_factory=list)


@dataclass
class GoalsConfig:
    thresholds: dict[str, float] = field(default_factory=dict)

    def get_threshold(self, metric: str, default: float) -> float:
        return float(self.thresholds.get(metric, default))


@dataclass
class LoopConfig:
    auto_seed: bool = False
    seed_backlog: Path = Path("seed/backlog.yaml")
    seed_config: Path | None = None
    seed_interval: float = 0.0
    seed_max_runs: int | None = None
    stop_when_idle: bool = True
    use_memory: bool = False
    memory_top_k: int = 3
    interactive: bool = False


@dataclass
class AuditConfig:
    enable_file_log: bool = True
    file_dir: Path = Path("seed/changes")
    enable_git_commit: bool = False
    commit_prefix: str = "A3X"


@dataclass
class ScalingConfig:
    cpu_threshold: float = 0.8
    memory_threshold: float = 0.8
    max_recursion_adjust: int = 3


@dataclass
class AgentConfig:
    llm: LLMConfig
    workspace: WorkspaceConfig
    limits: LimitsConfig
    tests: TestSettings
    policies: PoliciesConfig
    goals: GoalsConfig
    loop: LoopConfig
    audit: AuditConfig
    scaling: ScalingConfig = field(default_factory=ScalingConfig)

    @property
    def workspace_root(self) -> Path:
        return self.workspace.root.resolve()


def load_config(path: str | Path, validate: bool = True) -> AgentConfig:
    """
    Load and validate configuration from file.

    Args:
        path: Path to configuration file
        validate: Whether to validate configuration against schema

    Returns:
        Validated AgentConfig object

    Raises:
        ValidationError: If validation fails
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(path).resolve()

    # Validate configuration file if validation is enabled
    if validate and HAS_VALIDATION:
        try:
            validate_config_file(config_path, strict=True)
        except ImportError:
            # Validation framework not available, continue without validation
            pass

    data = _read_yaml(config_path)
    base_dir = config_path.parent

    llm_section = data.get("llm", {})
    script_value = llm_section.get("script")
    script_path = None
    if script_value is not None:
        script_path = (
            (base_dir / script_value).resolve()
            if not Path(script_value).is_absolute()
            else Path(script_value)
        )
    llm = LLMConfig(
        type=str(llm_section.get("type", "manual")),
        model=llm_section.get("model"),
        script=script_path,
        endpoint=llm_section.get("endpoint"),
        api_key_env=llm_section.get("api_key_env"),
        base_url=llm_section.get("base_url"),
    )

    workspace_section = data.get("workspace", {})
    root_value = workspace_section.get("root", ".")
    root_path = Path(root_value)
    if not root_path.is_absolute():
        root_path = (base_dir / root_path).resolve()
    workspace = WorkspaceConfig(
        root=root_path,
        allow_outside_root=bool(workspace_section.get("allow_outside_root", False)),
    )

    limits_section = data.get("limits", {})
    limits = LimitsConfig(
        max_iterations=int(limits_section.get("max_iterations", 50)),
        command_timeout=int(limits_section.get("command_timeout", 120)),
        max_failures=int(limits_section.get("max_failures", 10)),
        total_timeout=limits_section.get("total_timeout"),
    )

    tests_section = data.get("tests", {})
    raw_commands = tests_section.get("commands", [])
    commands: list[list[str]] = []
    for entry in raw_commands:
        if isinstance(entry, list):
            commands.append([str(part) for part in entry])
        elif isinstance(entry, str):
            commands.append(entry.split())
    tests = TestSettings(
        auto=bool(tests_section.get("auto", False)),
        commands=commands,
    )

    policies_section = data.get("policies", {})
    policies = PoliciesConfig(
        allow_network=bool(policies_section.get("allow_network", False)),
        allow_shell_write=bool(policies_section.get("allow_shell_write", True)),
        deny_commands=[str(item) for item in policies_section.get("deny_commands", [])],
    )

    goals_section = data.get("goals", {})
    goal_thresholds: dict[str, float] = {}
    for metric, spec in goals_section.items():
        if isinstance(spec, dict) and "min" in spec:
            goal_thresholds[str(metric)] = float(spec["min"])
        else:
            goal_thresholds[str(metric)] = float(spec)
    goals = GoalsConfig(thresholds=goal_thresholds)

    loop_section = data.get("loop", {})
    backlog_value = loop_section.get("seed_backlog", "seed/backlog.yaml")
    backlog_path = Path(backlog_value)
    if not backlog_path.is_absolute():
        backlog_path = (base_dir / backlog_path).resolve()
    seed_config_value = loop_section.get("seed_config")
    seed_config_path: Path | None = None
    if seed_config_value:
        seed_config_path = Path(seed_config_value)
        if not seed_config_path.is_absolute():
            seed_config_path = (base_dir / seed_config_path).resolve()
    loop = LoopConfig(
        auto_seed=bool(loop_section.get("auto_seed", False)),
        seed_backlog=backlog_path,
        seed_config=seed_config_path,
        seed_interval=float(loop_section.get("seed_interval", 0.0)),
        seed_max_runs=(
            int(loop_section["seed_max_runs"])
            if "seed_max_runs" in loop_section
            and loop_section["seed_max_runs"] is not None
            else None
        ),
        stop_when_idle=bool(loop_section.get("stop_when_idle", True)),
        use_memory=bool(loop_section.get("use_memory", False)),
        memory_top_k=max(0, int(loop_section.get("memory_top_k", 3))),
    )

    audit_section = data.get("audit", {})
    audit = AuditConfig(
        enable_file_log=bool(audit_section.get("enable_file_log", True)),
        file_dir=Path(audit_section.get("file_dir", "seed/changes")),
        enable_git_commit=bool(audit_section.get("enable_git_commit", False)),
        commit_prefix=str(audit_section.get("commit_prefix", "A3X")),
    )

    scaling_section = data.get("scaling", {})
    scaling = ScalingConfig(
        cpu_threshold=float(scaling_section.get("cpu_threshold", 0.8)),
        memory_threshold=float(scaling_section.get("memory_threshold", 0.8)),
        max_recursion_adjust=int(scaling_section.get("max_recursion_adjust", 3)),
    )

    return AgentConfig(
        llm=llm,
        workspace=workspace,
        limits=limits,
        tests=tests,
        policies=policies,
        goals=goals,
        loop=loop,
        audit=audit,
        scaling=scaling,
    )


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuração inválida: raiz deve ser um objeto")
    return data
