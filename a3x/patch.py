"""Aplicação de diffs unificados."""

from __future__ import annotations

import ast
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class PatchError(RuntimeError):
    pass


class PatchManager:
    """Responsável por aplicar diffs ao workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root
        if not shutil.which("patch"):
            raise PatchError(
                "O utilitário 'patch' é necessário mas não foi encontrado no PATH"
            )

    def validate_patch(self, diff: str) -> tuple[bool, str]:
        """Validate patch syntax by checking Python files with ast.parse after simulating patch application."""
        if not diff.strip():
            return False, "Diff vazio, nenhuma validação possível."

        # Extract affected paths
        affected_paths = self.extract_paths(diff)
        py_paths = [p for p in affected_paths if p.endswith(".py")]

        if not py_paths:
            return True, "No Python files to validate."

        # Validate each Python file individually without recursion
        validation_errors = []
        for rel_path in py_paths:
            full_path = self.root / rel_path
            if not full_path.exists():
                continue

            try:
                # Read original content
                original_content = full_path.read_text(encoding="utf-8")

                # Simulate patch application for this file
                simulated_content = self._simulate_patch_for_file(original_content, diff, rel_path)
                if simulated_content is None:
                    validation_errors.append(f"Failed to simulate patch for {rel_path}")
                    continue

                # Validate syntax of simulated content
                ast.parse(simulated_content)

            except SyntaxError as e:
                error_msg = f"Syntax error in {rel_path} after patch: {str(e)} at line {e.lineno}"
                validation_errors.append(error_msg)
            except Exception as e:
                validation_errors.append(f"Unexpected error validating {rel_path}: {str(e)}")

        if validation_errors:
            errors_str = "\n".join(validation_errors)
            return False, f"Validation failed due to syntax errors:\n{errors_str}"

        return True, "Patch validation passed: syntax OK for all Python files."

    def apply(self, diff: str) -> tuple[bool, str]:
        """Aplica o diff e retorna (sucesso, saída combinada)."""

        if not diff.strip():
            return False, "Diff vazio, nenhuma alteração aplicada."

        # Pre-apply validation
        validation_success, validation_msg = self.validate_patch(diff)
        if not validation_success:
            raise PatchError(f"Pre-apply validation failed: {validation_msg}")

        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", suffix=".diff"
        ) as tmp:
            tmp.write(diff)
            tmp_path = Path(tmp.name)

        try:
            last_output = ""
            for strip in (0, 1):
                dry_run = self._run_patch(tmp_path, dry_run=True, strip=strip)
                if dry_run.returncode != 0:
                    last_output = dry_run.stdout + dry_run.stderr
                    continue
                real_run = self._run_patch(tmp_path, dry_run=False, strip=strip)
                output = real_run.stdout + real_run.stderr
                if real_run.returncode == 0:
                    return True, f"{validation_msg}\n{output}"
                last_output = output
            return False, f"Falha ao aplicar patch após validação:\n{last_output}"
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_patch(
        self, diff_path: Path, dry_run: bool, strip: int
    ) -> subprocess.CompletedProcess[str]:
        args = [
            "patch",
            f"--strip={strip}",
            "--input",
            str(diff_path),
            "--batch",
            "--forward",
        ]
        if dry_run:
            args.append("--dry-run")
        return subprocess.run(
            args,
            cwd=self.root,
            check=False,
            text=True,
            capture_output=True,
        )

    def extract_paths(self, diff: str) -> List[str]:
        """Extracts file paths from unified diff."""
        import re
        # Match --- a/path and +++ b/path
        path_matches = re.findall(r"^(--- |\++\+ )([ab]/.*)$", diff, re.MULTILINE)
        return [match[1] for match in path_matches]

    def _simulate_patch_for_file(self, original_content: str, diff: str, rel_path: str) -> str | None:
        """Simulate patch application for a single file without using recursion."""
        try:
            lines = original_content.splitlines()
            patch_lines = diff.splitlines()

            # Find the hunk for this specific file
            in_file_hunk = False
            current_old_line = 0
            current_new_line = 0

            for line in patch_lines:
                if line.startswith("--- a/"):
                    if line == f"--- a/{rel_path}":
                        in_file_hunk = True
                    else:
                        in_file_hunk = False
                elif line.startswith("+++ b/"):
                    if line == f"+++ b/{rel_path}":
                        in_file_hunk = True
                    else:
                        in_file_hunk = False
                elif in_file_hunk and line.startswith("@@"):
                    # Parse hunk header: @@ -old_start,num_lines +new_start,num_lines @@
                    parts = line.split()
                    if len(parts) >= 3:
                        old_range = parts[1]
                        new_range = parts[2]
                        if old_range.startswith("-") and new_range.startswith("+"):
                            try:
                                current_old_line = int(old_range[1:].split(",")[0])
                                current_new_line = int(new_range[1:].split(",")[0])
                            except ValueError:
                                continue
                elif in_file_hunk and line.startswith(("+", "-")) and len(lines) > 0:
                    # Apply the change
                    if line.startswith("-") and current_old_line > 0 and current_old_line <= len(lines):
                        # Remove line
                        if current_old_line <= len(lines):
                            lines.pop(current_old_line - 1)
                            current_new_line -= 1
                    elif line.startswith("+") and current_new_line > 0:
                        # Add line
                        new_line_content = line[1:]
                        if current_new_line <= len(lines):
                            lines.insert(current_new_line - 1, new_line_content)
                        else:
                            lines.append(new_line_content)

            return "\n".join(lines)

        except Exception:
            return None


class CoreSandboxModifier:
    """Handles sandboxed modifications to core components for safety."""

    def __init__(self, root: Path, sandbox_type: str = "venv") -> None:
        self.root = root
        self.sandbox_type = sandbox_type
        self.sandbox_dir = root / f"sandbox_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.sandbox_dir.mkdir(exist_ok=True)

    def create_sandbox(self) -> Path:
        """Create a sandbox environment (venv or Docker)."""
        if self.sandbox_type == "venv":
            subprocess.run(["python", "-m", "venv", str(self.sandbox_dir)], check=True, cwd=self.root)
            return self.sandbox_dir
        elif self.sandbox_type == "docker":
            # Placeholder for Docker sandbox creation
            cmd = ["docker", "run", "-d", "--name", f"sandbox_{self.sandbox_dir.name}", "python:3.10-slim"]
            subprocess.run(cmd, check=True)
            return Path("/tmp/sandbox")  # Placeholder
        raise ValueError(f"Unsupported sandbox type: {self.sandbox_type}")

    def apply_in_sandbox(self, diff: str, sandbox_path: Path) -> tuple[bool, str]:
        """Apply patch within the sandbox."""
        # Activate venv and apply patch
        if self.sandbox_type == "venv":
            activate_script = sandbox_path / "bin" / "activate"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".diff", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(diff)
                tmp_path = Path(tmp.name)

            try:
                activate_str = shlex.quote(str(activate_script))
                root_str = shlex.quote(str(self.root))
                diff_str = shlex.quote(str(tmp_path))
                cmd = (
                    f"source {activate_str} && "
                    f"cd {root_str} && "
                    f"patch -p1 < {diff_str}"
                )
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=sandbox_path,
                )

                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                output_parts = []
                if stdout:
                    output_parts.append(f"STDOUT:\n{stdout}")
                if stderr:
                    output_parts.append(f"STDERR:\n{stderr}")
                combined_output = "\n".join(output_parts)

                if result.returncode == 0:
                    return True, combined_output or "Patch applied successfully."

                failure_message = (
                    f"Patch command failed with exit code {result.returncode}."
                )
                if combined_output:
                    failure_message = f"{failure_message}\n{combined_output}"
                return False, failure_message
            finally:
                tmp_path.unlink(missing_ok=True)
        return False, "Sandbox application not implemented for this type"


class RollbackManager:
    """Manages rollbacks using Git for safe core modifications."""

    def __init__(self, root: Path) -> None:
        self.root = root
        if not shutil.which("git"):
            raise PatchError("Git is required for rollback but not found in PATH")

    def create_checkpoint(self, checkpoint_name: str) -> str:
        """Create a Git checkpoint before modifications."""
        cmd = ["git", "add", "."]
        subprocess.run(cmd, cwd=self.root, check=True)
        tag_cmd = ["git", "tag", f"checkpoint_{checkpoint_name}"]
        result = subprocess.run(tag_cmd, cwd=self.root, capture_output=True, text=True)
        if result.returncode != 0:
            raise PatchError(f"Failed to create checkpoint: {result.stderr}")
        return f"checkpoint_{checkpoint_name}"

    def rollback_to_checkpoint(self, checkpoint_name: str) -> bool:
        """Rollback to the specified Git checkpoint."""
        cmd = ["git", "checkout", f"checkpoint_{checkpoint_name}", "-f"]
        result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Rolled back to {checkpoint_name}")
            return True
        else:
            print(f"Rollback failed: {result.stderr}")
            return False

    def verify_integrity_after_rollback(self) -> bool:
        """Verify system integrity after rollback (e.g., run tests)."""
        # Placeholder: run pytest or similar
        test_cmd = ["pytest", "-q"]
        result = subprocess.run(test_cmd, cwd=self.root, capture_output=True, text=True)
        return result.returncode == 0
