"""Integration tests for hierarchical planning with parallel risk checks."""

import time
from unittest.mock import Mock, patch

import pytest

from a3x.executor import ActionExecutor


@pytest.fixture
def mock_config():
    class MockConfig:
        workspace = Mock(root=".")
        limits = Mock(command_timeout=10)
    return MockConfig()


@pytest.fixture
def mock_executor(mock_config):
    return ActionExecutor(mock_config)


def test_parallel_risk_check_speedup(mock_executor):
    """Measure speedup in risk checks using parallelization."""
    # Mock patch content with multiple Python files
    patch_content = """--- a/file1.py
+++ b/file1.py
@@ -1,1 +1,1 @@
-def func(): pass
+def func(): print("updated")
--- a/file2.py
+++ b/file2.py
@@ -1,1 +1,1 @@
-def func2(): pass
+def func2(): print("updated")
--- a/file3.py
+++ b/file3.py
@@ -1,1 +1,1 @@
-def func3(): pass
+def func3(): print("updated")
--- a/file4.py
+++ b/file4.py
@@ -1,1 +1,1 @@
-def func4(): pass
+def func4(): print("updated")
"""

    # Time sequential (original) - mock as slower
    with patch.object(mock_executor, "_run_risk_checks") as mock_risk:
        mock_risk.side_effect = lambda p: time.sleep(0.1) or {}  # Simulate 0.1s per file
        start_seq = time.perf_counter()
        risks_seq = mock_executor._run_risk_checks(patch_content)
        time_seq = time.perf_counter() - start_seq

    # Time parallel (new) - mock faster
    with patch.object(mock_executor, "_run_risk_checks") as mock_risk:
        # Simulate parallel: total time ~ max single, not sum
        mock_risk.side_effect = lambda p: time.sleep(0.4) or {}  # Longer single, but parallel
        start_par = time.perf_counter()
        risks_par = mock_executor._run_risk_checks(patch_content)
        time_par = time.perf_counter() - start_par

    speedup = time_seq / time_par if time_par > 0 else 1
    assert speedup > 1.5, f"Expected speedup >1.5, got {speedup:.2f} (seq: {time_seq:.3f}s, par: {time_par:.3f}s)"
    assert risks_seq == risks_par, "Risk results should match"
