"""Tests for the PatchManager in A3X."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from hypothesis import given
from hypothesis.strategies import lists, text

from a3x.config import AgentConfig
from a3x.patch import PatchError, PatchManager


class TestPatchManager:
    """Tests for PatchManager."""

    def setup_method(self):
        """Setup test fixtures."""
        self.workspace_path = Path(tempfile.mkdtemp())
        self.config = AgentConfig(
            llm=Mock(),
            workspace=Mock(root=str(self.workspace_path)),
            limits=Mock(),
            tests=Mock(),
            policies=Mock(),
            goals=Mock(),
            loop=Mock(),
            audit=Mock()
        )
        self.pm = PatchManager(self.workspace_path)

    def teardown_method(self):
        """Teardown test fixtures."""
        import shutil
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)

    def test_patch_manager_initialization(self):
        """Test PatchManager initialization."""
        assert self.pm.root == self.workspace_path.resolve()
        # Verify patch command is available (system-dependent, but assume it is)
        import shutil
        assert shutil.which("patch") is not None, "patch utility required for tests"

    def test_apply_empty_diff(self):
        """Test applying empty diff."""
        success, output = self.pm.apply("")
        assert success is False
        assert "Diff vazio" in output

    def test_validate_patch_no_python_files(self):
        """Test validation with no Python files (should pass)."""
        diff = """
--- a/config.yaml
+++ b/config.yaml
@@ -1,1 +1,1 @@
-old: value
+new: value
"""
        success, msg = self.pm.validate_patch(diff)
        assert success is True
        assert "No Python files to validate" in msg

    def test_validate_patch_successful_syntax(self):
        """Test validation with valid Python syntax."""
        # Create a test Python file
        test_file = self.workspace_path / "test_valid.py"
        test_file.write_text('def valid_func():\n    return "ok"\n', encoding="utf-8")

        diff = """
--- a/test_valid.py
+++ b/test_valid.py
@@ -1,2 +1,2 @@
 def valid_func():
-    return "ok"
+    return "validated"
"""

        success, msg = self.pm.validate_patch(diff)
        assert success is True
        assert "validation passed" in msg.lower()
        assert "syntax OK" in msg

    def test_validate_patch_invalid_syntax(self):
        """Test validation detects invalid Python syntax and raises clear error."""
        # Create a test Python file
        test_file = self.workspace_path / "test_invalid.py"
        test_file.write_text('def valid_func():\n    return "ok"\n', encoding="utf-8")

        # Diff that introduces syntax error (incomplete string)
        invalid_diff = """
--- a/test_invalid.py
+++ b/test_invalid.py
@@ -1,2 +1,2 @@
 def valid_func():
-    return "ok"
+    return "incompl  # Missing closing quote - SyntaxError
"""

        success, msg = self.pm.validate_patch(invalid_diff)
        assert success is False
        assert "Syntax error" in msg
        assert "test_invalid.py" in msg
        assert "Missing closing quote" in msg or "invalid syntax" in msg.lower()
        assert "Validation failed due to syntax errors" in msg

    def test_apply_with_pre_validation_failure(self):
        """Test apply raises PatchError if pre-validation fails."""
        # Create test file
        test_file = self.workspace_path / "test_syntax.py"
        test_file.write_text("def func():\n    pass\n", encoding="utf-8")

        invalid_diff = """
--- a/test_syntax.py
+++ b/test_syntax.py
@@ -1,2 +1,2 @@
 def func():
-    pass
+    invalid_synt x  # Syntax error
"""

        with pytest.raises(PatchError, match="Pre-apply validation failed"):
            self.pm.apply(invalid_diff)

    def test_apply_successful_after_validation(self):
        """Test apply succeeds if validation passes."""
        # Create test file
        test_file = self.workspace_path / "test_apply.py"
        test_file.write_text("def func():\n    return True\n", encoding="utf-8")

        valid_diff = """
--- a/test_apply.py
+++ b/test_apply.py
@@ -1,2 +1,2 @@
 def func():
-    return True
+    return False  # Valid change
"""

        # Mock _run_patch to succeed
        with patch.object(self.pm, "_run_patch", return_value=Mock(returncode=0, stdout="Success", stderr="")):
            success, output = self.pm.apply(valid_diff)

        assert success is True
        assert "validation passed" in output
        assert "Success" in output

    def test_extract_paths_from_diff(self):
        """Test extracting paths from unified diff."""
        diff = """
--- a/a3x/agent.py
+++ b/a3x/agent.py
@@ -1,1 +1,1 @@
-old
+new

--- a/configs/sample.yaml
+++ b/configs/sample.yaml
@@ -1,1 +1,1 @@
-old config
+new config
"""
        paths = self.pm.extract_paths(diff)
        expected = ["a3x/agent.py", "configs/sample.yaml"]
        assert sorted(paths) == sorted(expected)

    def test_validate_patch_copy_failure(self):
        """Test validation handles copy failure gracefully."""
        diff = """
--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
-old
+new
"""
        # Mock file read to fail
        with patch("pathlib.Path.read_text", side_effect=Exception("Permission denied")):
            success, msg = self.pm.validate_patch(diff)

        assert success is False
        assert "Failed to copy" in msg


    @given(text(min_size=1))
    def test_robust_validate_patch_fuzz(self, random_diff):
        """Fuzz test for validate_patch robustness with random diff inputs."""
        try:
            success, msg = self.pm.validate_patch(random_diff)
            # Should either succeed or fail gracefully without crashing
            assert isinstance(success, bool)
            assert isinstance(msg, str)
        except Exception as e:
            # Only allow expected exceptions; raise if unexpected crash
            if not isinstance(e, (PatchError, ValueError, OSError)):
                raise AssertionError(f"Unexpected exception in validate_patch: {type(e).__name__}: {e}")


    @given(lists(text(min_size=1), min_size=1, max_size=5), lists(text(min_size=1), min_size=1, max_size=10))
    def test_robust_extract_paths_fuzz(self, paths, hunks):
        """Fuzz test for extract_paths robustness with random paths and hunks."""
        # Construct a simple diff with random paths and hunks
        diff_lines = []
        for path in paths:
            diff_lines.extend([
                f"--- a/{path}",
                f"+++ b/{path}",
                "@@ -1,1 +1,1 @@"
            ])
            for hunk in hunks[:2]:  # Limit hunks per path
                diff_lines.append(hunk[:50])  # Truncate for validity
            diff_lines.append("")  # Separator

        random_diff = "\n".join(diff_lines)
        try:
            extracted = self.pm.extract_paths(random_diff)
            # Should return list of paths, possibly empty or partial
            assert isinstance(extracted, list)
            for p in extracted:
                assert isinstance(p, str)
        except Exception as e:
            # Only allow expected exceptions
            if not isinstance(e, (ValueError, AttributeError)):
                raise AssertionError(f"Unexpected exception in extract_paths: {type(e).__name__}: {e}")
