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

    def extract_paths(self, diff: str) -> list[str]:
        """Extracts file paths from unified diff."""
        import re

        path_matches = re.findall(r"^(---|\+\+\+) ([ab]/[^\s]+)$", diff, re.MULTILINE)

        normalized_paths: list[str] = []
        for _, raw_path in path_matches:
            if raw_path.startswith(("a/", "b/")):
                normalized = raw_path[2:]
                if normalized not in normalized_paths:
                    normalized_paths.append(normalized)

        return normalized_paths

    def _simulate_patch_for_file(self, original_content: str, diff: str, rel_path: str) -> str | None:
        """Simulate patch application for a single file without using recursion."""

        def _parse_range(component: str) -> tuple[int, int]:
            value = component[1:]
            if "," in value:
                start_str, count_str = value.split(",", 1)
            else:
                start_str, count_str = value, "1"
            return int(start_str), int(count_str)

        try:
            patch_lines = diff.splitlines()
            hunks: list[tuple[str, list[str]]] = []
            current_file: str | None = None
            current_header = ""
            current_hunk: list[str] | None = None

            for line in patch_lines:
                if line.startswith("--- "):
                    if current_hunk is not None:
                        hunks.append((current_header, current_hunk))
                        current_hunk = None
                        current_header = ""
                    current_file = None
                elif line.startswith("+++ "):
                    if current_hunk is not None:
                        hunks.append((current_header, current_hunk))
                        current_hunk = None
                        current_header = ""
                    marker = line[4:]
                    normalized_marker = marker[2:] if marker.startswith("b/") else marker
                    if normalized_marker == rel_path:
                        current_file = rel_path
                    else:
                        current_file = None
                elif current_file == rel_path and line.startswith("@@"):
                    if current_hunk is not None:
                        hunks.append((current_header, current_hunk))
                    current_header = line
                    current_hunk = []
                elif current_file == rel_path and current_hunk is not None:
                    current_hunk.append(line)
                else:
                    if current_hunk is not None:
                        hunks.append((current_header, current_hunk))
                        current_hunk = None
                        current_header = ""

            if current_hunk is not None:
                hunks.append((current_header, current_hunk))

            if not hunks:
                return original_content

            original_lines = original_content.splitlines()
            result_lines: list[str] = []
            original_index = 0  # 0-based index into original_lines

            for header, hunk_lines in hunks:
                parts = header.split()
                if len(parts) < 3:
                    continue
                old_range = parts[1]
                old_start, _ = _parse_range(old_range)

                # Append unchanged lines before this hunk
                while original_index < old_start - 1 and original_index < len(original_lines):
                    result_lines.append(original_lines[original_index])
                    original_index += 1

                for hunk_line in hunk_lines:
                    if hunk_line.startswith(" "):
                        if original_index < len(original_lines):
                            result_lines.append(original_lines[original_index])
                        original_index += 1
                    elif hunk_line.startswith("-"):
                        original_index += 1
                    elif hunk_line.startswith("+"):
                        result_lines.append(hunk_line[1:])

            # Append the rest of the original content
            if original_index < len(original_lines):
                result_lines.extend(original_lines[original_index:])

            return "\n".join(result_lines)

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
