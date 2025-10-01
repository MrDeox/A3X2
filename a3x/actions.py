"""Definições de ações e observações do agente."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class ActionType(Enum):
    """Tipos de ações que o agente pode solicitar."""

    MESSAGE = auto()
    RUN_COMMAND = auto()
    APPLY_PATCH = auto()
    WRITE_FILE = auto()
    READ_FILE = auto()
    SELF_MODIFY = auto()
    FINISH = auto()
    # Data analysis actions
    ANALYZE_DATA = auto()
    VISUALIZE_DATA = auto()
    CLEAN_DATA = auto()
    STATISTICS = auto()


@dataclass
class AgentAction:
    """Descrição estruturada de uma ação solicitada pelo LLM."""

    type: ActionType
    text: Optional[str] = None
    command: Optional[List[str]] = None
    cwd: Optional[str] = None
    diff: Optional[str] = None
    path: Optional[str] = None
    content: Optional[str] = None
    dry_run: bool = False  # For SELF_MODIFY: whether to simulate without applying
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Observation:
    """Resultado de uma ação executada."""

    success: bool
    output: str = ""
    error: Optional[str] = None
    return_code: Optional[int] = None
    duration: float = 0.0
    type: str = "generic"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentState:
    """Estado enviado ao LLM para decidir a próxima ação."""

    goal: str
    history_snapshot: str
    iteration: int
    max_iterations: int
    seed_context: str = ""
    memory_lessons: str = ""
