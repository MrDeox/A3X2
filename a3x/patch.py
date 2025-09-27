"""Aplicação de diffs unificados."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class PatchError(RuntimeError):
    pass


class PatchManager:
    """Responsável por aplicar diffs ao workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root
        if not shutil.which("patch"):
            raise PatchError("O utilitário 'patch' é necessário mas não foi encontrado no PATH")

    def apply(self, diff: str) -> tuple[bool, str]:
        """Aplica o diff e retorna (sucesso, saída combinada)."""

        if not diff.strip():
            return False, "Diff vazio, nenhuma alteração aplicada."

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".diff") as tmp:
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
                    return True, output
                last_output = output
            return False, f"Falha ao aplicar patch:\n{last_output}"
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_patch(self, diff_path: Path, dry_run: bool, strip: int) -> subprocess.CompletedProcess[str]:
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
