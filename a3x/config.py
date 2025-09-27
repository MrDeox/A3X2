"""Carregamento e validação de configuração."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class LLMConfig:
    type: str
    model: Optional[str] = None
    script: Optional[Path] = None
    endpoint: Optional[str] = None
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class WorkspaceConfig:
    root: Path = Path('.')
    allow_outside_root: bool = False


@dataclass
class LimitsConfig:
    max_iterations: int = 50
    command_timeout: int = 120
    max_failures: int = 10
    total_timeout: Optional[int] = None


@dataclass
class TestsConfig:
    auto: bool = False
    commands: List[List[str]] = field(default_factory=list)


@dataclass
class PoliciesConfig:
    allow_network: bool = False
    allow_shell_write: bool = True
    deny_commands: List[str] = field(default_factory=list)


@dataclass
class GoalsConfig:
    thresholds: Dict[str, float] = field(default_factory=dict)

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


@dataclass
class AuditConfig:
    enable_file_log: bool = True
    file_dir: Path = Path("seed/changes")
    enable_git_commit: bool = False
    commit_prefix: str = "A3X"


@dataclass
class AgentConfig:
    llm: LLMConfig
    workspace: WorkspaceConfig
    limits: LimitsConfig
    tests: TestsConfig
    policies: PoliciesConfig
    goals: GoalsConfig
    loop: LoopConfig
    audit: AuditConfig

    @property
    def workspace_root(self) -> Path:
        return self.workspace.root.resolve()


def load_config(path: str | Path) -> AgentConfig:
    config_path = Path(path).resolve()
    data = _read_yaml(config_path)
    base_dir = config_path.parent

    llm_section = data.get("llm", {})
    script_value = llm_section.get("script")
    script_path = None
    if script_value is not None:
        script_path = (base_dir / script_value).resolve() if not Path(script_value).is_absolute() else Path(script_value)
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
    commands: List[List[str]] = []
    for entry in raw_commands:
        if isinstance(entry, list):
            commands.append([str(part) for part in entry])
        elif isinstance(entry, str):
            commands.append(entry.split())
    tests = TestsConfig(
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
    goal_thresholds: Dict[str, float] = {}
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
        seed_max_runs=int(loop_section["seed_max_runs"]) if "seed_max_runs" in loop_section and loop_section["seed_max_runs"] is not None else None,
        stop_when_idle=bool(loop_section.get("stop_when_idle", True)),
    )

    audit_section = data.get("audit", {})
    audit = AuditConfig(
        enable_file_log=bool(audit_section.get("enable_file_log", True)),
        file_dir=Path(audit_section.get("file_dir", "seed/changes")),
        enable_git_commit=bool(audit_section.get("enable_git_commit", False)),
        commit_prefix=str(audit_section.get("commit_prefix", "A3X")),
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
    )


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuração inválida: raiz deve ser um objeto")
    return data
