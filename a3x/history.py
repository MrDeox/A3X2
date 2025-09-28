"""Histórico e contexto do agente."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .actions import AgentAction, Observation


@dataclass
class HistoryEvent:
    """Combinação de ação executada e observação recebida."""

    action: AgentAction
    observation: Observation


class AgentHistory:
    """Mantém registro sequencial das interações do agente."""

    def __init__(self) -> None:
        self._events: List[HistoryEvent] = []

    def append(self, action: AgentAction, observation: Observation) -> None:
        self._events.append(HistoryEvent(action=action, observation=observation))

    @property
    def events(self) -> List[HistoryEvent]:
        return list(self._events)

    def snapshot(self, max_chars: int = 12_000) -> str:
        """Retorna versão textual compacta do histórico."""

        lines: List[str] = []
        for idx, event in enumerate(self._events, start=1):
            action_desc = _describe_action(event.action)
            obs_desc = _describe_observation(event.observation)
            lines.append(f"[{idx}] ACTION: {action_desc}")
            if obs_desc:
                lines.append(f"[{idx}] OBS   : {obs_desc}")
            if sum(len(line) for line in lines) > max_chars:
                lines.append("... (histórico truncado) ...")
                break
        return "\n".join(lines)


def _describe_action(action: AgentAction) -> str:
    if action.type.name == "MESSAGE":
        return action.text or "(mensagem vazia)"
    if action.type.name == "RUN_COMMAND":
        cmd = " ".join(action.command or [])
        return f"RUN `{cmd}` (cwd={action.cwd or '.'})"
    if action.type.name == "APPLY_PATCH":
        if not action.diff:
            return "APPLY_PATCH (sem diff)"
        return f"APPLY_PATCH ({len(action.diff.splitlines())} linhas de diff)"
    if action.type.name == "WRITE_FILE":
        return f"WRITE_FILE {action.path} ({len(action.content or '')} chars)"
    if action.type.name == "READ_FILE":
        return f"READ_FILE {action.path}"
    if action.type.name == "FINISH":
        return action.text or "(fim)"
    return action.type.name


def _describe_observation(obs: Observation) -> str:
    prefix = "OK" if obs.success else "FAIL"
    details = []
    if obs.return_code is not None:
        details.append(f"code={obs.return_code}")
    if obs.duration:
        details.append(f"t={obs.duration:.2f}s")
    if obs.output:
        excerpt = obs.output.strip().splitlines()[:3]
        joined = " ".join(line.strip() for line in excerpt)
        if len(joined) > 160:
            joined = joined[:157] + "..."
        details.append(joined)
    if obs.error:
        err_excerpt = obs.error.strip().splitlines()[:2]
        details.append("ERR: " + " ".join(err_excerpt))
    return f"{prefix} ({', '.join(details)})" if details else prefix
