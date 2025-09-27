"""Executa ações solicitadas pelo agente."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

from .actions import ActionType, AgentAction, Observation
from .config import AgentConfig
from .patch import PatchManager, PatchError
from .change_log import ChangeLogger


class ActionExecutor:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.workspace_root = config.workspace_root
        self.patch_manager = PatchManager(self.workspace_root)
        self.change_logger = ChangeLogger(
            self.workspace_root,
            enable_file_log=config.audit.enable_file_log,
            file_dir=config.audit.file_dir,
            enable_git_commit=config.audit.enable_git_commit,
            commit_prefix=config.audit.commit_prefix,
        )

    def execute(self, action: AgentAction) -> Observation:
        handler_name = f"_handle_{action.type.name.lower()}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            return Observation(success=False, output="", error=f"Ação não suportada: {action.type}")
        return handler(action)

    # Handlers -----------------------------------------------------------------

    def _handle_message(self, action: AgentAction) -> Observation:
        return Observation(success=True, output=action.text or "", type="message")

    def _handle_finish(self, action: AgentAction) -> Observation:
        return Observation(success=True, output=action.text or "", type="finish")

    def _handle_read_file(self, action: AgentAction) -> Observation:
        if not action.path:
            return Observation(success=False, error="Caminho não informado", type="read_file")
        target = self._resolve_workspace_path(action.path)
        if not target.exists():
            return Observation(success=False, error=f"Arquivo não encontrado: {target}", type="read_file")
        try:
            content = target.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - leitura raramente falha
            return Observation(success=False, error=str(exc), type="read_file")
        return Observation(success=True, output=content, type="read_file")

    def _handle_write_file(self, action: AgentAction) -> Observation:
        if not action.path:
            return Observation(success=False, error="Caminho não informado", type="write_file")
        target = self._resolve_workspace_path(action.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = ""
        if target.exists():
            try:
                before = target.read_text(encoding="utf-8")
            except Exception:
                before = ""
        try:
            target.write_text(action.content or "", encoding="utf-8")
        except Exception as exc:
            return Observation(success=False, error=str(exc), type="write_file")
        try:
            self.change_logger.log_write(target, before, action.content or "", note="write_file")
        except Exception:
            pass
        return Observation(success=True, output=f"Escrito {target}", type="write_file")

    def _handle_apply_patch(self, action: AgentAction) -> Observation:
        if not action.diff:
            return Observation(success=False, error="Diff vazio", type="apply_patch")
        try:
            success, output = self.patch_manager.apply(action.diff)
            if success:
                try:
                    self.change_logger.log_patch(action.diff, note="apply_patch")
                except Exception:
                    pass
            return Observation(success=success, output=output, type="apply_patch")
        except PatchError as exc:
            return Observation(success=False, error=str(exc), type="apply_patch")

    def _handle_run_command(self, action: AgentAction) -> Observation:
        if not action.command:
            return Observation(success=False, error="Comando não informado", type="run_command")

        if not self._command_allowed(action.command):
            return Observation(success=False, error="Comando bloqueado por política", type="run_command")

        cwd = self._resolve_cwd(action.cwd)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                action.command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.config.limits.command_timeout,
                env=self._build_env(),
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return Observation(
                success=False,
                output=exc.stdout or "",
                error=f"Timeout após {self.config.limits.command_timeout}s",
                return_code=None,
                duration=duration,
                type="run_command",
            )
        except FileNotFoundError:
            duration = time.perf_counter() - start
            joined = " ".join(shlex.quote(part) for part in action.command)
            return Observation(
                success=False,
                output="",
                error=f"Executável não encontrado para comando: {joined}",
                return_code=None,
                duration=duration,
                type="run_command",
            )

        duration = time.perf_counter() - start
        output = (proc.stdout or "")
        error = proc.stderr or None
        success = proc.returncode == 0
        return Observation(
            success=success,
            output=output,
            error=error,
            return_code=proc.returncode,
            duration=duration,
            type="run_command",
        )

    # Helpers ------------------------------------------------------------------

    def _resolve_cwd(self, cwd: Optional[str]) -> Path:
        if cwd:
            return self._resolve_workspace_path(cwd)
        return self.workspace_root

    def _resolve_workspace_path(self, path: str) -> Path:
        candidate = (self.workspace_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if not self.config.workspace.allow_outside_root and self.workspace_root not in candidate.parents and candidate != self.workspace_root:
            raise PermissionError(f"Acesso negado fora do workspace: {candidate}")
        return candidate

    def _command_allowed(self, command: list[str]) -> bool:
        joined = " ".join(command)
        for pattern in self.config.policies.deny_commands:
            if pattern in joined:
                return False
        return True

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if not self.config.policies.allow_network:
            env.setdefault("NO_NETWORK", "1")
        return env
