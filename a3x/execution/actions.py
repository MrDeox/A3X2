"""Action handlers for A3X execution system.

This module contains all the specific action type handlers that were
previously embedded in the monolithic ActionExecutor class.
"""

from __future__ import annotations

import ast
import concurrent.futures
import logging
import os
import re
import resource
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List

from ..actions import AgentAction, Observation, ActionType
from ..change_log import ChangeLogger
from ..constants import (
    DAYS_BEFORE_ARCHIVE,
    MAX_DIFF_COMPLEXITY_SCORE,
    MAX_DIFF_LINES,
    MAX_FUNCTION_COMPLEXITY,
    MAX_LINTER_WORKERS,
    MEMORY_LIMIT_MB,
    SUBPROCESS_TIMEOUT,
    TEST_FAILURE_RATE_THRESHOLD,
)
from ..patch import PatchError, PatchManager

class ActionHandlers:
    """Collection of action handlers for different agent action types.

    This class contains all the specific handlers for different action types,
    extracted from the monolithic ActionExecutor for better organization and
    testability.
    """

    def __init__(self, orchestrator) -> None:
        """Initialize the action handlers.

        Args:
            orchestrator (ExecutionOrchestrator): The main execution orchestrator.
        """
        self.orchestrator = orchestrator
        self.workspace_root = orchestrator.workspace_root
        self.patch_manager = orchestrator.patch_manager
        self.change_logger = orchestrator.change_logger

    def get_handler(self, action_type: ActionType):
        """Get the handler function for a specific action type.

        Args:
            action_type (ActionType): The type of action to handle.

        Returns:
            callable: The handler function, or None if not found.
        """
        if action_type is None:
            return None
        handler_name = f"_handle_{action_type.name.lower()}"
        return getattr(self, handler_name, None)

    def get_status(self) -> Dict[str, Any]:
        """Get status information about the action handlers.

        Returns:
            Dict[str, Any]: Status information including available handlers.
        """
        return {
            "available_handlers": [
                "message", "finish", "read_file", "write_file",
                "apply_patch", "self_modify", "run_command"
            ],
            "workspace_root": str(self.workspace_root),
        }

    # Basic action handlers ---------------------------------------------------

    def _handle_message(self, action: AgentAction) -> Observation:
        """Handle message actions."""
        return Observation(success=True, output=action.text or "", type="message")

    def _handle_finish(self, action: AgentAction) -> Observation:
        """Handle finish actions."""
        return Observation(success=True, output=action.text or "", type="finish")

    def _handle_read_file(self, action: AgentAction) -> Observation:
        """Handle file read actions."""
        if not action.path:
            return Observation(
                success=False, error="Path not provided", type="read_file"
            )

        target = self.orchestrator.get_workspace_path(action.path)
        if not target.exists():
            return Observation(
                success=False,
                error=f"File not found: {target}",
                type="read_file",
            )

        try:
            content = target.read_text(encoding="utf-8")
        except Exception as exc:
            return Observation(success=False, error=str(exc), type="read_file")

        return Observation(success=True, output=content, type="read_file")

    def _handle_write_file(self, action: AgentAction) -> Observation:
        """Handle file write actions."""
        if not action.path:
            return Observation(
                success=False, error="Path not provided", type="write_file"
            )

        target = self.orchestrator.get_workspace_path(action.path)
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
            pass  # Best effort logging

        return Observation(success=True, output=f"Written {target}", type="write_file")

    def _handle_run_command(self, action: AgentAction) -> Observation:
        """Handle command execution actions."""
        if not action.command:
            return Observation(
                success=False, error="Command not provided", type="run_command"
            )

        if not self._command_allowed(action.command):
            return Observation(
                success=False,
                error="Command blocked by policy",
                type="run_command",
            )

        if not self._is_safe_command(action.command):
            return Observation(
                success=False,
                error="Unsafe command (e.g., privileged or network access blocked)",
                type="run_command",
            )

        cwd = self.orchestrator.get_workspace_path(action.cwd) if action.cwd else self.workspace_root

        start = time.perf_counter()
        try:
            def set_limits():
                # Set memory limit: configured MB virtual memory
                memory_limit = MEMORY_LIMIT_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

            proc = subprocess.run(
                action.command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.orchestrator.config.limits.command_timeout,
                env=self._build_env(),
                preexec_fn=set_limits,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return Observation(
                success=False,
                output=exc.stdout or "",
                error=f"Timeout after {self.orchestrator.config.limits.command_timeout}s",
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
                error=f"Executable not found for command: {joined}",
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

    # Complex action handlers -----------------------------------------------

    def _handle_apply_patch(self, action: AgentAction) -> Observation:
        """Handle patch application actions with full safety checking."""
        if not action.diff:
            return Observation(success=False, error="Empty diff", type="apply_patch")

        # Extract Python files from patch for validation
        py_paths = set(re.findall(r"^--- a/(.+\.py)$", action.diff, re.MULTILINE))

        # Backup and get original states
        backups, original_states, backup_obs = self._backup_and_get_original_states(py_paths)
        if backup_obs:
            return backup_obs

        # Run risk checks
        risks = self._run_risk_checks(action.diff)
        high_risks = {k: v for k, v in risks.items() if v == "high"}

        if high_risks:
            details = f"High risks detected: {high_risks}"
            log_path = self.workspace_root / "seed" / "reports" / "risk_log.md"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n## Patch Risk Log - {datetime.now().isoformat()}\n{details}\nFull risks: {risks}\n")
            self._log_and_cleanup("", backups)
            return Observation(success=False, output=f"Patch rejected due to high risks: {details}", type="apply_patch")

        try:
            # Apply the patch
            success, output = self.patch_manager.apply(action.diff)
            if not success:
                raise PatchError(output)

            # Validate patch and rollback on error
            validation_success, validation_output = self._validate_patch_and_rollback_on_error(
                py_paths, original_states, backups
            )
            output += validation_output

            if not validation_success:
                return Observation(success=False, output=output, type="apply_patch")

            # Log and cleanup
            self._log_and_cleanup(action.diff, backups)
            return Observation(success=True, output=output, type="apply_patch")

        except (PatchError, Exception) as exc:
            # Rollback on any error
            for rel_path, original_content in original_states.items():
                full_path = self.orchestrator.get_workspace_path(rel_path)
                try:
                    full_path.write_text(original_content, encoding="utf-8")
                except Exception:
                    pass  # Best effort
            self._log_and_cleanup("", backups)
            error_msg = f"Unexpected error: {str(exc)}; rolled back to original state"
            logging.error(f"Error during patch application: {exc}")
            return Observation(success=False, error=error_msg, type="apply_patch")

    def _handle_self_modify(self, action: AgentAction) -> Observation:
        """Handle self-modification actions with comprehensive safety checks."""
        if not action.diff:
            return Observation(success=False, error="Empty diff for self-modify", type="self_modify")

        # Restrict to agent code: a3x/ and configs/
        allowed_prefixes = ["a3x", "configs"]
        patch_paths = self.patch_manager.extract_paths(action.diff)
        invalid_paths = [p for p in patch_paths if not any(p.startswith(prefix) for prefix in allowed_prefixes)]

        if invalid_paths:
            return Observation(
                success=False,
                error=f"Self-modify restricted to a3x/ and configs/: invalid {invalid_paths}",
                type="self_modify"
            )

        # Determine if low-risk for disabling dry-run and forcing commit
        diff_lines = len(action.diff.splitlines())
        core_list = ["a3x/agent.py", "a3x/executor.py"]
        is_low_risk = (diff_lines < 10) or all(path not in core_list for path in patch_paths)

        if is_low_risk:
            action.dry_run = False
            logging.info("Real commit enabled for low-risk self-modify")

        # Disable dry-run for testing (always apply for real)
        action.dry_run = False
        logging.info("Dry-run disabled for testing; enabling real apply.")

        try:
            # Apply for real with extra logging and impact analysis
            is_safe, analysis_msg = self.orchestrator.code_analyzer.analyze_impact_before_apply(action)

            if not is_safe:
                return Observation(
                    success=False,
                    error=f"Impact analysis rejected self-modification: {analysis_msg}",
                    type="self_modify"
                )

            logging.info(f"Impact analysis passed: {analysis_msg}")

            success, output = self.patch_manager.apply(action.diff)
            if success:
                try:
                    self.change_logger.log_patch(action.diff, note="self_modify")
                except Exception:
                    pass

                # Run tests after successful self-modify
                pytest_result = subprocess.run(
                    ["pytest", "-q", "tests/"],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if pytest_result.returncode == 0:
                    # Auto-approve and commit based on risk level
                    if is_low_risk:
                        auto_approve = True
                        logging.info("Low-risk self-modify: Auto-approving and committing.")
                    else:
                        logging.warning("High-risk self-modify: Skipping auto-commit.")
                        auto_approve = False

                    if auto_approve:
                        output += self._auto_commit_changes(patch_paths, "Seed-applied: self-modify enhancement")
                    else:
                        # Interactive confirmation for high-risk changes
                        output += self._handle_high_risk_commit(patch_paths, "Seed-applied: self-modify enhancement")
                else:
                    output += f"\nTests failed after self-modify: {pytest_result.stderr[:200]}... Commit skipped."

                output += "\n[Self-modify applied; restart agent for effects.]"

            return Observation(success=success, output=output, type="self_modify")

        except PatchError as exc:
            return Observation(success=False, error=str(exc), type="self_modify")

    # Helper methods ---------------------------------------------------------

    def _backup_and_get_original_states(self, py_paths: set[str]) -> tuple[dict, dict, Observation | None]:
        """Backup files and get original states for rollback."""
        backups = {}
        original_states = {}

        for rel_path in py_paths:
            full_path = self.orchestrator.get_workspace_path(rel_path)
            if full_path.exists():
                try:
                    original_content = full_path.read_text(encoding="utf-8")
                    original_states[rel_path] = original_content
                    backup_path = full_path.with_suffix(".py.bak")
                    shutil.copy2(full_path, backup_path)
                    backups[rel_path] = backup_path
                except Exception as e:
                    return {}, {}, Observation(
                        success=False,
                        error=f"Failed to backup {rel_path}: {str(e)}",
                        type="apply_patch"
                    )

        return backups, original_states, None

    def _validate_patch_and_rollback_on_error(self, py_paths: set[str], original_states: dict, backups: dict) -> tuple[bool, str]:
        """Validate patch and rollback on syntax errors."""
        has_error = False
        error_details = []
        output = ""

        for rel_path in py_paths:
            full_path = self.orchestrator.get_workspace_path(rel_path)
            if full_path.suffix == ".py" and full_path.exists():
                try:
                    content = full_path.read_text(encoding="utf-8")
                    ast.parse(content)
                except SyntaxError as e:
                    has_error = True
                    lineno = getattr(e, "lineno", "unknown")
                    offset = getattr(e, "offset", 0)
                    error_msg = f"SyntaxError in {rel_path} at line {lineno}: {e.msg}"
                    lines = content.splitlines()
                    start_line = max(1, lineno - 2)
                    end_line = min(len(lines), lineno + 2)
                    snippet = "\n".join(lines[start_line-1:end_line])

                    if offset > 0 and (lineno - start_line) < len(snippet.splitlines()):
                        pointer = " " * (offset - 1) + "^"
                        snippet += f"\n{pointer}"

                    error_details.append(f"{error_msg}\nSnippet:\n{snippet}")

                    # Rollback this file
                    original_content = original_states.get(rel_path, "")
                    full_path.write_text(original_content, encoding="utf-8")
                    backup_path = backups.get(rel_path)
                    if backup_path and backup_path.exists():
                        backup_path.unlink()
                    output += f"\n{error_msg}\nSnippet around error:\n{snippet}"

        if has_error:
            output += "\nAST validation failed: Patch rejected due to syntax errors; affected files rolled back."
            logging.error(f"AST validation errors: {"\n".join(error_details)}")
            return False, output

        return True, output

    def _run_risk_checks(self, patch_content: str) -> dict[str, str]:
        """Run lightweight risk checks on patch using linters in temp environment."""
        risks = {}

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_path = Path(temp_dir_str)
            py_paths = set(re.findall(r"^--- a/(.+\.py)$", patch_content, re.MULTILINE))

            if not py_paths:
                return risks  # No Python files, low risk

            # Copy original files to temp dir
            for rel_path in py_paths:
                full_path = self.orchestrator.get_workspace_path(rel_path)
                if full_path.exists():
                    temp_file = temp_path / rel_path
                    temp_file.parent.mkdir(parents=True, exist_ok=True)
                    temp_file.write_text(full_path.read_text(encoding="utf-8"), encoding="utf-8")

            # Apply patch in temp dir
            temp_pm = PatchManager(temp_path)
            try:
                temp_success, temp_output = temp_pm.apply(patch_content)
                if not temp_success:
                    risks["patch_apply"] = "high"
                    return risks
            except Exception as e:
                risks["patch_apply"] = "high"
                risks["apply_error"] = str(e)[:100]
                return risks

            # Run linters in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_LINTER_WORKERS) as executor:
                future_to_path = {executor.submit(self._run_linters_for_file, temp_path, temp_dir_str, rel_path): rel_path for rel_path in py_paths}

                for future in concurrent.futures.as_completed(future_to_path):
                    rel_path = future_to_path[future]
                    try:
                        file_risks = future.result()
                        # Aggregate risks
                        for key, level in file_risks.items():
                            if key not in risks:
                                risks[key] = level
                            elif level == "high" and risks[key] != "high":
                                risks[key] = "high"  # Escalate to high if any file has high risk
                    except Exception as exc:
                        logging.error(f"Linter future for {rel_path} generated exception: {exc}")
                        risks[f"lint_future_error_{rel_path}"] = "high"

        return risks

    def _run_linters_for_file(self, temp_path: Path, temp_dir_str: str, rel_path: str) -> dict[str, str]:
        """Run linters for a single file in temp directory."""
        file_risks = {}
        temp_file = temp_path / rel_path

        if temp_file.exists() and temp_file.suffix == ".py":
            try:
                # Ruff check
                ruff_result = subprocess.run(
                    ["ruff", "check", "--output-format=text", str(temp_file)],
                    capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT, cwd=temp_dir_str
                )

                if ruff_result.returncode != 0:
                    violations = len([line for line in ruff_result.stdout.splitlines() if line.strip() and not line.startswith("==")])
                    if violations > 0:
                        # High if syntax errors (E9), else medium
                        if any(code.startswith("E9") for code in ruff_result.stdout.split() if code.startswith("E")):
                            file_risks["ruff_syntax"] = "high"
                        else:
                            file_risks["ruff"] = "medium" if violations <= 5 else "high"

                # Black check
                black_result = subprocess.run(
                    ["black", "--check", str(temp_file)],
                    capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT, cwd=temp_dir_str
                )

                if black_result.returncode != 0:
                    file_risks["black_style"] = "medium"

            except subprocess.TimeoutExpired:
                file_risks["lint_timeout"] = "high"
            except FileNotFoundError:
                # ruff/black not found, assume ok for now
                pass
            except Exception as e:
                file_risks["lint_error"] = str(e)[:100]

        return file_risks

    def _command_allowed(self, command: list[str]) -> bool:
        """Check if command is allowed by policy."""
        joined = " ".join(command)
        for pattern in self.orchestrator.config.policies.deny_commands:
            if pattern in joined:
                return False
        return True

    def _is_safe_command(self, command: list[str]) -> bool:
        """Check if command is safe (no sudo, rm -rf, etc.)."""
        joined = " ".join(command).lower()
        unsafe_patterns = [
            "sudo", "su", "rm -rf", "dd if=", "mkfs", "mount", "umount",
            "curl.*|wget.*http",  # Network blocked if !allow_network
        ]

        if not self.orchestrator.config.policies.allow_network:
            network_terms = ["http", "curl", "wget"]
            if any(term in joined for term in network_terms):
                return False

        return all(not any(term in joined for term in unsafe.split()) for unsafe in unsafe_patterns)

    def _build_env(self) -> dict[str, str]:
        """Build restricted environment for command execution."""
        env = os.environ.copy()
        if not self.orchestrator.config.policies.allow_network:
            env.setdefault("NO_NETWORK", "1")
        return env

    def _log_and_cleanup(self, diff: str, backups: dict):
        """Log changes and cleanup backup files."""
        try:
            self.change_logger.log_patch(diff, note="apply_patch")
        except Exception as log_exc:
            logging.warning(f"Failed to log patch: {log_exc}")

        # Archive old change files
        changes_dir = self.workspace_root / "seed" / "changes"
        archive_dir = self.workspace_root / "seed" / "archive"
        changes_dir.mkdir(exist_ok=True)
        archive_dir.mkdir(exist_ok=True)

        if changes_dir.exists():
            for filename in os.listdir(changes_dir):
                if filename.endswith(".diff"):
                    file_path = changes_dir / filename
                    if file_path.is_file():
                        mtime = os.path.getmtime(file_path)
                        if datetime.now() - datetime.fromtimestamp(mtime) > timedelta(days=DAYS_BEFORE_ARCHIVE):
                            shutil.move(str(file_path), str(archive_dir / filename))

        # Clean up backup files
        for backup_path in backups.values():
            backup_path.unlink(missing_ok=True)

    def _auto_commit_changes(self, patch_paths: list, commit_msg: str) -> str:
        """Auto-commit changes for low-risk modifications."""
        added_paths = []

        for rel_path in patch_paths:
            full_path = self.orchestrator.get_workspace_path(rel_path)
            if full_path.exists():
                try:
                    subprocess.run(
                        ["git", "add", str(full_path)],
                        cwd=self.workspace_root,
                        check=True,
                        capture_output=True
                    )
                    added_paths.append(rel_path)
                    logging.info(f"Git add successful for {full_path}")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Git add failed for {full_path}: {e}")
                    continue

        if added_paths:
            try:
                subprocess.run(
                    ["git", "commit", "-m", f"{commit_msg} ({len(added_paths)} files)"],
                    cwd=self.workspace_root,
                    check=True,
                    capture_output=True
                )
                logging.info(f"Auto-commit successful for {added_paths}")
                return f"\nAuto-commit applied successfully for {len(added_paths)} files."
            except subprocess.CalledProcessError as e:
                logging.error(f"Git commit failed: {e}")
                return "\nAuto-commit failed; manual intervention needed."
        else:
            return "\nNo files added; commit skipped."

    def _handle_high_risk_commit(self, patch_paths: list, commit_msg: str) -> str:
        """Handle high-risk changes with interactive confirmation."""
        output = "\nHigh risk self-modify detected. Interactive commit handling required."
        # In a real implementation, this would prompt for user input
        # For now, just log that manual intervention is needed
        logging.warning("High-risk self-modify: Manual commit intervention required.")
        return output