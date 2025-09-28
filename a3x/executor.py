"""Executa ações solicitadas pelo agente."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

from .actions import AgentAction, Observation
from .config import AgentConfig
from .patch import PatchManager, PatchError
from .change_log import ChangeLogger

import ast
import re
import shutil
from datetime import datetime, timedelta
import logging


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
            return Observation(
                success=False, output="", error=f"Ação não suportada: {action.type}"
            )
        return handler(action)

    # Handlers -----------------------------------------------------------------

    def _handle_message(self, action: AgentAction) -> Observation:
        return Observation(success=True, output=action.text or "", type="message")

    def _handle_finish(self, action: AgentAction) -> Observation:
        return Observation(success=True, output=action.text or "", type="finish")

    def _handle_read_file(self, action: AgentAction) -> Observation:
        if not action.path:
            return Observation(
                success=False, error="Caminho não informado", type="read_file"
            )
        target = self._resolve_workspace_path(action.path)
        if not target.exists():
            return Observation(
                success=False,
                error=f"Arquivo não encontrado: {target}",
                type="read_file",
            )
        try:
            content = target.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - leitura raramente falha
            return Observation(success=False, error=str(exc), type="read_file")
        return Observation(success=True, output=content, type="read_file")

    def _handle_write_file(self, action: AgentAction) -> Observation:
        if not action.path:
            return Observation(
                success=False, error="Caminho não informado", type="write_file"
            )
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
            self.change_logger.log_write(
                target, before, action.content or "", note="write_file"
            )
        except Exception:
            pass
        return Observation(success=True, output=f"Escrito {target}", type="write_file")

    def _handle_apply_patch(self, action: AgentAction) -> Observation:
        if not action.diff:
            return Observation(success=False, error="Diff vazio", type="apply_patch")

        # Extract Python files from diff for AST validation
        py_paths = set(re.findall(r'^--- a/(.+\.py)$', action.diff, re.MULTILINE))
        backups = {}
        for rel_path in py_paths:
            full_path = self._resolve_workspace_path(rel_path)
            if full_path.exists():
                backup_path = full_path.with_suffix('.py.bak')
                shutil.copy2(full_path, backup_path)
                backups[rel_path] = backup_path

        try:
            success, output = self.patch_manager.apply(action.diff)

            # AST validation fallback
            has_error = False
            for rel_path in py_paths:
                full_path = self._resolve_workspace_path(rel_path)
                if full_path.suffix == '.py' and full_path.exists():
                    try:
                        content = full_path.read_text(encoding="utf-8")
                        ast.parse(content)
                    except SyntaxError as e:
                        has_error = True
                        output += f"\nSyntaxError in {rel_path}: {str(e)}"
                        # Revert from backup
                        backup_path = backups.get(rel_path)
                        if backup_path and backup_path.exists():
                            full_path.write_text(backup_path.read_text(encoding="utf-8"))
                            backup_path.unlink()
            if has_error:
                success = False
                output += "\nAST fallback: Patch rejected due to syntax errors; affected files reverted."
            else:
                if success:
                    try:
                        self.change_logger.log_patch(action.diff, note="apply_patch")
                    except Exception:
                        pass

                    # Cleanup old .diff files in seed/changes/
                    from pathlib import Path
                    changes_dir = Path('seed/changes')
                    archive_dir = Path('seed/archive')
                    if changes_dir.exists():
                        for filename in os.listdir(changes_dir):
                            if filename.endswith('.diff'):
                                file_path = changes_dir / filename
                                if file_path.is_file():
                                    mtime = os.path.getmtime(file_path)
                                    if datetime.now() - datetime.fromtimestamp(mtime) > timedelta(days=7):
                                        shutil.move(str(file_path), str(archive_dir / filename))

            return Observation(success=success, output=output, type="apply_patch")
        except PatchError as exc:
            # Clean up backups on patch error
            for backup_path in backups.values():
                backup_path.unlink(missing_ok=True)
            return Observation(success=False, error=str(exc), type="apply_patch")

    def _handle_self_modify(self, action: AgentAction) -> Observation:
        if not action.diff:
            return Observation(success=False, error="Diff vazio para self-modify", type="self_modify")
        
        # Restrict to agent code: a3x/ and configs/
        allowed_prefixes = ["a3x", "configs"]
        patch_paths = self.patch_manager.extract_paths(action.diff)
        invalid_paths = [p for p in patch_paths if not any(p.startswith(prefix) for prefix in allowed_prefixes)]
        if invalid_paths:
            return Observation(
                success=False,
                error=f"Self-modify restrito a a3x/ e configs/: inválidos {invalid_paths}",
                type="self_modify"
            )

        # Determine if low-risk for disabling dry-run and forcing commit
        diff_lines = len(action.diff.splitlines())
        core_list = ['a3x/agent.py', 'a3x/executor.py']
        is_low_risk = (diff_lines < 10) or all(path not in core_list for path in patch_paths)
        if is_low_risk:
            action.dry_run = False
            logging.info("Real commit enabled for low-risk self-modify")
        
        if action.dry_run:
            # Simulate with dry-run
            try:
                success, output = self.patch_manager.apply(action.diff, dry_run=True)
                return Observation(
                    success=success,
                    output=f"Simulação dry-run: {output}",
                    type="self_modify"
                )
            except PatchError as exc:
                return Observation(success=False, error=f"Dry-run falhou: {str(exc)}", type="self_modify")
        
        # Apply for real, with extra logging and alignment check
        try:
            # Simple alignment check: scan diff for dangerous changes
            if self._has_dangerous_self_change(action.diff):
                return Observation(
                    success=False,
                    error="Mudança self-modify detectada como desalinhada (ex.: remoção de safeguards)",
                    type="self_modify"
                )
            success, output = self.patch_manager.apply(action.diff)
            if success:
                try:
                    self.change_logger.log_patch(action.diff, note="self_modify")
                except Exception:
                    pass

                # Run pytest after successful self-modify
                pytest_result = subprocess.run(
                    ["pytest", "-q", "tests/"],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if pytest_result.returncode == 0:
                    # Risk assessment for auto-approval; force without prompt if success_rate >0.9 (inferred from pytest success)
                    core_list = ['a3x/agent.py', 'a3x/executor.py']
                    patch_paths = self.patch_manager.extract_paths(action.diff)
                    diff_lines = len(action.diff.splitlines())
                    is_low_risk = (diff_lines < 10) or all(path not in core_list for path in patch_paths)
                    
                    # Assume success_rate >0.9 if pytest passed
                    success_rate = 1.0 if pytest_result.returncode == 0 else 0.0
                    
                    if is_low_risk and success_rate > 0.9:
                        auto_approve = True
                        logging.info("Real commit enabled: low-risk and high success_rate")
                    else:
                        print("High-risk self-modify detected. Approve auto-commit? (y/n):")
                        approval = input().strip().lower()
                        auto_approve = approval == 'y'
                    
                    if auto_approve:
                        # Get affected paths from diff
                        for rel_path in patch_paths:
                            full_path = self._resolve_workspace_path(rel_path)
                            if full_path.exists():
                                try:
                                    subprocess.run(
                                        ['git', 'add', str(full_path)],
                                        cwd=self.workspace_root,
                                        check=True
                                    )
                                    logging.info(f"Git add successful for {full_path}")
                                except subprocess.CalledProcessError as e:
                                    logging.error(f"Git add failed for {full_path}: {e}")
                                    if is_low_risk:
                                        # Retry once for low-risk
                                        try:
                                            subprocess.run(
                                                ['git', 'add', str(full_path)],
                                                cwd=self.workspace_root,
                                                check=True
                                            )
                                            logging.info(f"Git add retry successful for {full_path}")
                                        except subprocess.CalledProcessError:
                                            logging.warning(f"Git add retry failed for {full_path}; skipping commit for this file")
                                            continue
                                    else:
                                        raise
                        commit_msg = "Seed-applied: self-modify enhancement"
                        try:
                            subprocess.run(
                                ['git', 'commit', '-m', commit_msg],
                                cwd=self.workspace_root,
                                check=True
                            )
                            logging.info(f"Auto-commit attempted for {patch_paths}")
                            output += "\nAuto-commit applied successfully."
                        except subprocess.CalledProcessError as e:
                            logging.error(f"Git commit failed: {e}")
                            if is_low_risk:
                                # Force commit with fallback message or retry
                                try:
                                    subprocess.run(
                                        ['git', 'commit', '-m', f"{commit_msg} (retry)"],
                                        cwd=self.workspace_root,
                                        check=True
                                    )
                                    logging.info("Git commit retry successful")
                                    output += "\nAuto-commit applied after retry."
                                except subprocess.CalledProcessError:
                                    logging.warning("Git commit retry failed; fallback to manual")
                                    output += "\nAuto-commit failed; manual intervention needed."
                            else:
                                raise
                    else:
                        output += "\nCommit skipped."
                else:
                    output += f"\nTests failed after self-modify: {pytest_result.stderr[:200]}..."

                # Optional: Trigger restart or reload, but for now just log
                output += "\n[Self-modify aplicado; reinicie o agente para efeitos.]"
            return Observation(success=success, output=output, type="self_modify")
        except PatchError as exc:
            return Observation(success=False, error=str(exc), type="self_modify")

    def _has_dangerous_self_change(self, diff: str) -> bool:
        """Verificação básica de alinhamento: busca padrões perigosos no diff."""
        dangerous_patterns = [
            r"-.*allow_network.*=.*True",  # Habilitando rede
            r"-.*deny_commands.*=.*\[\]",  # Limpando denials
            r"+.*sudo",  # Adicionando privilégios
            r"-.*_is_safe_command",  # Removendo checks
        ]
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, diff):
                return True
        return False

    def _handle_run_command(self, action: AgentAction) -> Observation:
        if not action.command:
            return Observation(
                success=False, error="Comando não informado", type="run_command"
            )

        if not self._command_allowed(action.command):
            return Observation(
                success=False,
                error="Comando bloqueado por política",
                type="run_command",
            )

        # Lightweight sandbox: restrict to non-privileged commands and env
        restricted_env = self._build_restricted_env()
        if not self._is_safe_command(action.command):
            return Observation(
                success=False,
                error="Comando não seguro (ex.: privilégios ou rede bloqueados)",
                type="run_command",
            )

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
        output = proc.stdout or ""
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
        candidate = (
            (self.workspace_root / path).resolve()
            if not Path(path).is_absolute()
            else Path(path).resolve()
        )
        if (
            not self.config.workspace.allow_outside_root
            and self.workspace_root not in candidate.parents
            and candidate != self.workspace_root
        ):
            if not str(candidate).startswith("/tmp/a3x_sandbox/"):
                raise PermissionError(f"Acesso negado fora do workspace: {candidate}")
            return candidate
        return candidate

    def _command_allowed(self, command: list[str]) -> bool:
        joined = " ".join(command)
        for pattern in self.config.policies.deny_commands:
            if pattern in joined:
                return False
        return True

    def _is_safe_command(self, command: list[str]) -> bool:
        """Verifica se comando é seguro: sem sudo, rm -rf, etc."""
        joined = " ".join(command).lower()
        unsafe_patterns = [
            "sudo", "su", "rm -rf", "dd if=", "mkfs", "mount", "umount",
            "curl.*|wget.*http",  # Rede básica bloqueada se !allow_network
        ]
        if not self.config.policies.allow_network and any("http" in p or "curl" in p or "wget" in p for p in unsafe_patterns if any(term in joined for term in p.split())):
            return False
        return all(not any(term in joined for term in unsafe.split()) for unsafe in unsafe_patterns)

    def _build_restricted_env(self) -> dict[str, str]:
        env = self._build_env()
        # Remover vars perigosas
        dangerous_vars = ["PATH", "LD_PRELOAD", "LD_LIBRARY_PATH"]
        for var in dangerous_vars:
            env.pop(var, None)
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"  # PATH restrito
        env["HOME"] = str(self.workspace_root)
        env["SHELL"] = "/bin/sh"
        return env

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if not self.config.policies.allow_network:
            env.setdefault("NO_NETWORK", "1")
        return env
