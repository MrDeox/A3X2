"""Change logging utilities: archives diffs and optionally commits to git."""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class ChangeLogger:
    def __init__(
        self,
        root: Path,
        *,
        enable_file_log: bool = True,
        file_dir: Path | str = Path("seed/changes"),
        enable_git_commit: bool = False,
        commit_prefix: str = "A3X",
    ) -> None:
        self.root = Path(root)
        self.enable_file_log = enable_file_log
        self.file_dir = (self.root / file_dir).resolve() if not Path(file_dir).is_absolute() else Path(file_dir)
        self.enable_git_commit = enable_git_commit
        self.commit_prefix = commit_prefix
        if self.enable_file_log:
            self.file_dir.mkdir(parents=True, exist_ok=True)

    def log_patch(self, diff_text: str, *, note: str = "") -> None:
        if self.enable_file_log and diff_text.strip():
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            path = self.file_dir / f"{ts}_{int(time.time()*1000)%1000:03d}_apply_patch.diff"
            path.write_text(diff_text, encoding="utf-8")
        if self.enable_git_commit and diff_text.strip():
            self._git_commit(message=f"{self.commit_prefix}: apply_patch {note}".strip())

    def log_write(self, path: Path, before: str, after: str, *, note: str = "") -> None:
        # Build a minimal unified diff header
        rel = path.relative_to(self.root) if path.is_absolute() else path
        header = f"--- a/{rel}\n+++ b/{rel}\n"
        import difflib

        diff_lines = difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile=str(rel), tofile=str(rel)
        )
        diff_text = header + "".join(diff_lines)
        if self.enable_file_log and diff_text.strip():
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            out = self.file_dir / f"{ts}_{int(time.time()*1000)%1000:03d}_write_file.diff"
            out.write_text(diff_text, encoding="utf-8")
        if self.enable_git_commit:
            self._git_commit(message=f"{self.commit_prefix}: write_file {rel} {note}".strip())

    def _git_commit(self, *, message: str) -> None:
        try:
            # only if inside a git repository
            if not (self.root / ".git").exists():
                return
            subprocess.run(["git", "add", "-A"], cwd=self.root, check=False, capture_output=True)
            subprocess.run(["git", "commit", "-m", message], cwd=self.root, check=False, capture_output=True)
        except Exception:
            pass


__all__ = ["ChangeLogger"]

