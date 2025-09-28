"""Tests for the ActionExecutor class."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import re

from a3x.executor import ActionExecutor, Observation
from a3x.config import AgentConfig, LLMConfig, WorkspaceConfig, LimitsConfig, TestsConfig, PoliciesConfig, GoalsConfig, LoopConfig, AuditConfig
from a3x.actions import AgentAction
from a3x.patch import PatchManager


@pytest.fixture
def config():
    return AgentConfig(
        llm=LLMConfig(type="manual"),
        workspace=WorkspaceConfig(root=Path(".")),
        limits=LimitsConfig(command_timeout=10),
        tests=TestsConfig(),
        policies=PoliciesConfig(deny_commands=[], allow_network=False),
        goals=GoalsConfig(),
        loop=LoopConfig(),
        audit=AuditConfig(enable_file_log=False, enable_git_commit=False),
    )


@pytest.fixture
def executor(config):
    executor = ActionExecutor(config)
    executor.patch_manager = Mock()
    executor.change_logger = Mock()
    return executor


def test_handle_apply_patch_success(executor):
    diff = """--- a/test.py
 +++ b/test.py
 @@ -1,0 +1,1 @@
 +print("Hello World")
 """
    py_paths = ['test.py']
    with patch('a3x.executor.re.findall', return_value=py_paths):
        executor.patch_manager.apply.return_value = (True, "Patch applied")
        with patch.object(executor, '_resolve_workspace_path') as mock_resolve:
            mock_file = Mock()
            mock_file.suffix = '.py'
            mock_file.exists.return_value = True
            mock_file.read_text.return_value = 'print("Hello World")\n'
            mock_resolve.return_value = mock_file
            with patch('a3x.executor.ast.parse', return_value=None):
                with patch('a3x.executor.shutil.copy2'):
                    action = AgentAction(type="apply_patch", diff=diff)
                    obs = executor._handle_apply_patch(action)
    assert obs.success
    assert "applied" in obs.output.lower()


def test_handle_apply_patch_syntax_error_revert(executor):
    diff = """--- a/test.py
+++ b/test.py
@@ -1,0 +1,1 @@
+print("unclosed
"""
    py_paths = ['test.py']
    with patch('a3x.executor.re.findall', return_value=py_paths):
        executor.patch_manager.apply.return_value = (True, "Patch applied")
        with patch.object(executor, '_resolve_workspace_path') as mock_resolve:
            mock_file = Mock()
            mock_file.suffix = '.py'
            mock_file.exists.return_value = True
            mock_file.read_text.return_value = 'print("unclosed\n'
            mock_backup = Mock()
            mock_backup.exists.return_value = True
            mock_backup.read_text.return_value = "original valid"
            mock_backup.unlink = Mock()
            mock_file.with_suffix.return_value = mock_backup
            mock_file.write_text = Mock()
            mock_resolve.return_value = mock_file
            with patch('a3x.executor.shutil.copy2'):
                with patch('a3x.executor.ast.parse', side_effect=SyntaxError("Invalid")):
                    action = AgentAction(type="apply_patch", diff=diff)
                    obs = executor._handle_apply_patch(action)
    assert not obs.success
    assert "SyntaxError" in obs.output
    assert "AST fallback" in obs.output
    assert "reverted" in obs.output.lower()


def test_handle_apply_patch_non_python(executor):
    diff = """--- a/test.txt
+++ b/test.txt
@@ -1,0 +1,1 @@
+World
"""
    py_paths = []  # No .py files
    with patch('a3x.executor.re.findall', return_value=py_paths):
        executor.patch_manager.apply.return_value = (True, "Patch applied")
        action = AgentAction(type="apply_patch", diff=diff)
        obs = executor._handle_apply_patch(action)
    assert obs.success
    # No AST check


def test_ast_fallback_improves_success_rate(executor):
    # Simulate two applies: one success, one failure detected by AST
    diff_good = """--- a/good.py
+++ b/good.py
@@ -1,0 +1,1 @@
+print("ok")
"""
    diff_bad = """--- a/bad.py
+++ b/bad.py
@@ -1,0 +1,1 @@
+def func print("bad")
"""
    py_paths_good = ['good.py']
    py_paths_bad = ['bad.py']
    with patch('a3x.executor.re.findall', side_effect=[py_paths_good, py_paths_bad]):
        executor.patch_manager.apply.return_value = (True, "Patch applied")
        with patch.object(executor, '_resolve_workspace_path') as mock_resolve:
            mock_good = Mock()
            mock_good.suffix = '.py'
            mock_good.exists.return_value = True
            mock_good.read_text.return_value = 'print("ok")\n'
            mock_bad = Mock()
            mock_bad.suffix = '.py'
            mock_bad.exists.return_value = True
            mock_bad.read_text.return_value = 'def func print("bad")\n'
            mock_backup_good = Mock()
            mock_backup_good.exists.return_value = False
            mock_good.with_suffix.return_value = mock_backup_good
            mock_good.write_text = Mock()

            mock_backup_bad = Mock()
            mock_backup_bad.exists.return_value = True
            mock_backup_bad.read_text.return_value = "original"
            mock_backup_bad.unlink = Mock()
            mock_bad.with_suffix.return_value = mock_backup_bad
            mock_bad.write_text = Mock()

            mock_resolve.side_effect = [mock_good, mock_good, mock_bad, mock_bad]
            with patch('a3x.executor.ast.parse', side_effect=[None, SyntaxError()]):
                with patch('a3x.executor.shutil.copy2'):
                    # Good
                    action_good = AgentAction(type="apply_patch", diff=diff_good)
                    obs_good = executor._handle_apply_patch(action_good)
                    assert obs_good.success

                    # Bad, fallback
                    action_bad = AgentAction(type="apply_patch", diff=diff_bad)
                    obs_bad = executor._handle_apply_patch(action_bad)
                    assert not obs_bad.success
                    assert "SyntaxError" in obs_bad.output
                    assert "AST fallback" in obs_bad.output
    # Fallback improves rate by detecting and reverting invalid patches
def test_handle_self_modify_with_auto_commit(executor):
    diff = """--- a/a3x/test.py
+++ b/a3x/test.py
@@ -0,0 +1 @@
+print("self modify test")
"""
    executor.patch_manager.extract_paths.return_value = ['a3x/test.py']
    executor.patch_manager.apply.return_value = (True, "Applied")
    mock_pytest = Mock()
    mock_pytest.returncode = 0
    mock_git_add = Mock()
    mock_git_add.returncode = 0
    mock_git_commit = Mock()
    mock_git_commit.returncode = 0
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.__str__.return_value = 'a3x/test.py'
    with patch('subprocess.run', side_effect=[mock_pytest, mock_git_add, mock_git_commit]) as mock_run:
        with patch('builtins.input', return_value='y'):
            with patch.object(executor, '_resolve_workspace_path', return_value=mock_path):
                with patch.object(executor, '_has_dangerous_self_change', return_value=False):
                    action = AgentAction(type="self_modify", diff=diff)
                    obs = executor._handle_self_modify(action)
    assert obs.success
    assert "Auto-commit applied successfully." in obs.output
    mock_run.assert_any_call(["pytest", "-q", "tests/"], cwd=executor.workspace_root, capture_output=True, text=True, timeout=30)
    mock_run.assert_any_call(['git', 'add', 'a3x/test.py'], cwd=executor.workspace_root, check=True)
    mock_run.assert_any_call(['git', 'commit', '-m', "Seed-applied: self-modify enhancement"], cwd=executor.workspace_root, check=True)


def test_handle_self_modify_skip_commit(executor):
    diff = """--- a/a3x/test.py
+++ b/a3x/test.py
@@ -0,0 +1 @@
+print("self modify test")
"""
    executor.patch_manager.extract_paths.return_value = ['a3x/test.py']
    executor.patch_manager.apply.return_value = (True, "Applied")
    mock_pytest = Mock()
    mock_pytest.returncode = 0
    with patch('subprocess.run', return_value=mock_pytest) as mock_run:
        with patch('builtins.input', return_value='n'):
            with patch.object(executor, '_resolve_workspace_path', return_value=Path('a3x/test.py')):
                with patch.object(executor, '_has_dangerous_self_change', return_value=False):
                    action = AgentAction(type="self_modify", diff=diff)
                    obs = executor._handle_self_modify(action)
    assert obs.success
    assert "Commit skipped by user." in obs.output
    mock_run.assert_called_once_with(["pytest", "-q", "tests/"], cwd=executor.workspace_root, capture_output=True, text=True, timeout=30)


def test_handle_self_modify_tests_fail(executor):
    diff = """--- a/a3x/test.py
+++ b/a3x/test.py
@@ -0,0 +1 @@
+print("self modify test")
"""
    executor.patch_manager.extract_paths.return_value = ['a3x/test.py']
    executor.patch_manager.apply.return_value = (True, "Applied")
    mock_pytest = Mock()
    mock_pytest.returncode = 1
    mock_pytest.stderr = "Test failed"
    with patch('subprocess.run', return_value=mock_pytest) as mock_run:
        with patch.object(executor, '_resolve_workspace_path', return_value=Path('a3x/test.py')):
            with patch.object(executor, '_has_dangerous_self_change', return_value=False):
                action = AgentAction(type="self_modify", diff=diff)
                obs = executor._handle_self_modify(action)
    assert obs.success
    assert "Tests failed after self-modify" in obs.output
    mock_run.assert_called_once_with(["pytest", "-q", "tests/"], cwd=executor.workspace_root, capture_output=True, text=True, timeout=30)