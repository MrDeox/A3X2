"""Comprehensive tests for the change_log module."""

import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from a3x.change_log import ChangeLogger


class TestChangeLogger:
    """Test cases for the ChangeLogger class."""

    def test_logger_creation_minimal(self) -> None:
        """Test creating logger with minimal configuration."""
        with patch("a3x.change_log.Path.mkdir"):
            logger = ChangeLogger(root=Path("/test"))

            assert logger.root == Path("/test")
            assert logger.enable_file_log is True
            assert logger.file_dir == Path("/test/seed/changes")
            assert logger.enable_git_commit is False
            assert logger.commit_prefix == "A3X"
            assert logger.file_dir.exists()  # mkdir should have been called

    def test_logger_creation_full_config(self) -> None:
        """Test creating logger with full configuration."""
        with patch("a3x.change_log.Path.mkdir"):
            logger = ChangeLogger(
                root=Path("/custom/root"),
                enable_file_log=False,
                file_dir="custom/changes",
                enable_git_commit=True,
                commit_prefix="CUSTOM"
            )

            assert logger.root == Path("/custom/root")
            assert logger.enable_file_log is False
            assert logger.file_dir == Path("/custom/root/custom/changes")
            assert logger.enable_git_commit is True
            assert logger.commit_prefix == "CUSTOM"

    def test_logger_creation_relative_file_dir(self) -> None:
        """Test creating logger with relative file directory."""
        with patch("a3x.change_log.Path.mkdir"):
            logger = ChangeLogger(
                root=Path("/test/root"),
                file_dir="relative/changes"
            )

            assert logger.file_dir == Path("/test/root/relative/changes")

    def test_logger_creation_absolute_file_dir(self) -> None:
        """Test creating logger with absolute file directory."""
        with patch("a3x.change_log.Path.mkdir"):
            logger = ChangeLogger(
                root=Path("/test/root"),
                file_dir="/absolute/changes"
            )

            assert logger.file_dir == Path("/absolute/changes")

    def test_logger_initializes_directory(self) -> None:
        """Test that logger creates directory if file logging is enabled."""
        with patch("a3x.change_log.Path.mkdir") as mock_mkdir:
            ChangeLogger(root=Path("/test"))

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_logger_no_directory_creation_when_disabled(self) -> None:
        """Test that logger doesn't create directory when file logging is disabled."""
        with patch("a3x.change_log.Path.mkdir") as mock_mkdir:
            ChangeLogger(
                root=Path("/test"),
                enable_file_log=False
            )

            mock_mkdir.assert_not_called()


class TestLogPatch:
    """Test cases for the log_patch method."""

    def test_log_patch_with_file_logging_enabled(self) -> None:
        """Test logging patch with file logging enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            diff_text = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def test():
     pass
+    # Added new line
"""

            with patch("a3x.change_log.datetime") as mock_datetime:
                mock_datetime.now.return_value = datetime(2023, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
                with patch("time.time", return_value=1673776245.123):

                    logger.log_patch(diff_text, note="test patch")

                    # Check that file was created
                    expected_filename = "20230115-103045_123_apply_patch.diff"
                    expected_path = logger.file_dir / expected_filename
                    assert expected_path.exists()

                    # Check file content
                    content = expected_path.read_text(encoding='utf-8')
                    assert diff_text in content

    def test_log_patch_with_empty_diff(self) -> None:
        """Test logging patch with empty diff."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            # Empty diff should not create file
            logger.log_patch("", note="empty patch")

            # Check no files were created
            assert len(list(logger.file_dir.glob("*"))) == 0

    def test_log_patch_with_whitespace_only_diff(self) -> None:
        """Test logging patch with whitespace-only diff."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            # Whitespace-only diff should not create file
            logger.log_patch("   \n\t  \n  ", note="whitespace patch")

            # Check no files were created
            assert len(list(logger.file_dir.glob("*"))) == 0

    def test_log_patch_with_file_logging_disabled(self) -> None:
        """Test logging patch with file logging disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=False)

            diff_text = "test diff content"

            logger.log_patch(diff_text, note="disabled logging")

            # Check no files were created
            assert len(list(logger.file_dir.glob("*"))) == 0

    def test_log_patch_with_git_commit_enabled(self) -> None:
        """Test logging patch with git commit enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_path, check=True, capture_output=True)

            logger = ChangeLogger(root=temp_path, enable_git_commit=True, commit_prefix="TEST")

            diff_text = "test diff"

            with patch("subprocess.run") as mock_subprocess:
                logger.log_patch(diff_text, note="test commit")

                # Should call git add and git commit
                assert mock_subprocess.call_count == 2
                mock_subprocess.assert_any_call(
                    ["git", "add", "-A"], cwd=temp_path, check=False, capture_output=True
                )
                mock_subprocess.assert_any_call(
                    ["git", "commit", "-m", "TEST: apply_patch test commit"],
                    cwd=temp_path, check=False, capture_output=True
                )

    def test_log_patch_with_git_commit_no_repo(self) -> None:
        """Test logging patch with git commit enabled but no git repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_git_commit=True)

            diff_text = "test diff"

            with patch("subprocess.run") as mock_subprocess:
                logger.log_patch(diff_text, note="no repo")

                # Should not call git commands if .git doesn't exist
                mock_subprocess.assert_not_called()

    def test_log_patch_filename_format(self) -> None:
        """Test that patch log filename follows correct format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            diff_text = "test diff content"

            with patch("a3x.change_log.datetime") as mock_datetime:
                mock_datetime.now.return_value = datetime(2023, 12, 25, 14, 30, 45, tzinfo=timezone.utc)
                with patch("time.time", return_value=1703512245.987):

                    logger.log_patch(diff_text)

                    # Check filename format: YYYYMMDD-HHMMSS_MMM_apply_patch.diff
                    files = list(logger.file_dir.glob("*"))
                    assert len(files) == 1

                    filename = files[0].name
                    assert filename.startswith("20231225-143045_987_")
                    assert filename.endswith("_apply_patch.diff")


class TestLogWrite:
    """Test cases for the log_write method."""

    def test_log_write_with_file_logging_enabled(self) -> None:
        """Test logging file write with file logging enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            before_content = "line 1\nline 2\nline 3"
            after_content = "line 1\nline 2\nmodified line 3\nline 4"

            with patch("a3x.change_log.datetime") as mock_datetime:
                mock_datetime.now.return_value = datetime(2023, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
                with patch("time.time", return_value=1673776245.456):

                    logger.log_write(
                        path=Path("test/file.py"),
                        before=before_content,
                        after=after_content,
                        note="test write"
                    )

                    # Check that diff file was created
                    expected_filename = "20230115-103045_456_write_file.diff"
                    expected_path = logger.file_dir / expected_filename
                    assert expected_path.exists()

                    # Check file content contains unified diff format
                    content = expected_path.read_text(encoding='utf-8')
                    assert "--- a/test/file.py" in content
                    assert "+++ b/test/file.py" in content
                    assert "+modified line 3" in content
                    assert "+line 4" in content
                    assert "-line 3" in content

    def test_log_write_with_file_logging_disabled(self) -> None:
        """Test logging file write with file logging disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=False)

            logger.log_write(
                path=Path("test.py"),
                before="old content",
                after="new content"
            )

            # Check no files were created
            assert len(list(logger.file_dir.glob("*"))) == 0

    def test_log_write_with_git_commit_enabled(self) -> None:
        """Test logging file write with git commit enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_path, check=True, capture_output=True)

            logger = ChangeLogger(root=temp_path, enable_git_commit=True, commit_prefix="TEST")

            with patch("subprocess.run") as mock_subprocess:
                logger.log_write(
                    path=Path("src/main.py"),
                    before="print('hello')",
                    after="print('hello world')",
                    note="add world"
                )

                # Should call git add and git commit
                assert mock_subprocess.call_count == 2
                mock_subprocess.assert_any_call(
                    ["git", "add", "-A"], cwd=temp_path, check=False, capture_output=True
                )
                mock_subprocess.assert_any_call(
                    ["git", "commit", "-m", "TEST: write_file src/main.py add world"],
                    cwd=temp_path, check=False, capture_output=True
                )

    def test_log_write_with_absolute_path(self) -> None:
        """Test logging file write with absolute path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            absolute_file_path = temp_path / "absolute" / "test.py"

            logger.log_write(
                path=absolute_file_path,
                before="old",
                after="new"
            )

            # Check that diff file was created with correct relative path in diff
            files = list(logger.file_dir.glob("*"))
            assert len(files) == 1

            content = files[0].read_text(encoding='utf-8')
            assert "--- a/absolute/test.py" in content
            assert "+++ b/absolute/test.py" in content

    def test_log_write_with_relative_path(self) -> None:
        """Test logging file write with relative path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            relative_file_path = Path("relative/test.py")

            logger.log_write(
                path=relative_file_path,
                before="old",
                after="new"
            )

            # Check that diff file was created with correct relative path in diff
            files = list(logger.file_dir.glob("*"))
            assert len(files) == 1

            content = files[0].read_text(encoding='utf-8')
            assert "--- a/relative/test.py" in content
            assert "+++ b/relative/test.py" in content

    def test_log_write_identical_content(self) -> None:
        """Test logging file write with identical before and after content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            content = "identical content"

            logger.log_write(
                path=Path("test.py"),
                before=content,
                after=content
            )

            # Should not create file for identical content
            assert len(list(logger.file_dir.glob("*"))) == 0

    def test_log_write_empty_content(self) -> None:
        """Test logging file write with empty content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            logger.log_write(
                path=Path("empty.py"),
                before="",
                after=""
            )

            # Should not create file for empty identical content
            assert len(list(logger.file_dir.glob("*"))) == 0

    def test_log_write_with_note_injection(self) -> None:
        """Test that note is properly included in git commit message."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_path, check=True, capture_output=True)

            logger = ChangeLogger(root=temp_path, enable_git_commit=True)

            with patch("subprocess.run") as mock_subprocess:
                logger.log_write(
                    path=Path("test.py"),
                    before="old",
                    after="new",
                    note="important fix"
                )

                # Check that note is included in commit message
                commit_call = None
                for call in mock_subprocess.call_args_list:
                    if call[0][0] == ["git", "commit", "-m", "A3X: write_file test.py important fix"]:
                        commit_call = call
                        break

                assert commit_call is not None

    def test_log_write_filename_format(self) -> None:
        """Test that write log filename follows correct format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            with patch("a3x.change_log.datetime") as mock_datetime:
                mock_datetime.now.return_value = datetime(2023, 6, 15, 9, 15, 30, tzinfo=timezone.utc)
                with patch("time.time", return_value=1686815730.555):

                    logger.log_write(
                        path=Path("test.py"),
                        before="old",
                        after="new"
                    )

                    # Check filename format: YYYYMMDD-HHMMSS_MMM_write_file.diff
                    files = list(logger.file_dir.glob("*"))
                    assert len(files) == 1

                    filename = files[0].name
                    assert filename.startswith("20230615-091530_555_")
                    assert filename.endswith("_write_file.diff")


class TestGitIntegration:
    """Test cases for git integration functionality."""

    def test_git_commit_with_existing_repo(self) -> None:
        """Test git commit when repository exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Initialize git repo and create initial commit
            subprocess.run(["git", "init"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_path, check=True, capture_output=True)
            (temp_path / "initial.txt").write_text("initial content")
            subprocess.run(["git", "add", "initial.txt"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_path, check=True, capture_output=True)

            logger = ChangeLogger(root=temp_path, enable_git_commit=True)

            with patch("subprocess.run") as mock_subprocess:
                logger.log_patch("test diff", note="test")

                # Should call git add and git commit
                mock_subprocess.assert_any_call(
                    ["git", "add", "-A"], cwd=temp_path, check=False, capture_output=True
                )
                mock_subprocess.assert_any_call(
                    ["git", "commit", "-m", "A3X: apply_patch test"],
                    cwd=temp_path, check=False, capture_output=True
                )

    def test_git_commit_handles_subprocess_errors(self) -> None:
        """Test that git commit handles subprocess errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_path, check=True, capture_output=True)

            logger = ChangeLogger(root=temp_path, enable_git_commit=True)

            # Mock subprocess to raise exception
            with patch("subprocess.run", side_effect=Exception("Git error")):
                # Should not raise exception, just continue
                logger.log_patch("test diff", note="should not fail")

    def test_git_commit_with_no_changes(self) -> None:
        """Test git commit when there are no changes to commit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Initialize git repo with initial commit
            subprocess.run(["git", "init"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "commit", "--allow-empty", "-m", "Initial"], cwd=temp_path, check=True, capture_output=True)

            logger = ChangeLogger(root=temp_path, enable_git_commit=True)

            with patch("subprocess.run") as mock_subprocess:
                # Log something that won't create actual changes
                logger.log_patch("")

                # Should still call git add, but git commit might fail silently
                mock_subprocess.assert_called_with(
                    ["git", "add", "-A"], cwd=temp_path, check=False, capture_output=True
                )


class TestChangeLoggerIntegration:
    """Integration tests for change logger."""

    def test_full_logging_workflow(self) -> None:
        """Test complete logging workflow with both patches and file writes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            # Log a patch
            patch_diff = """--- a/main.py
+++ b/main.py
@@ -1,3 +1,4 @@
 def main():
     print("Hello")
+    print("World")
"""

            logger.log_patch(patch_diff, note="add greeting")

            # Log a file write
            logger.log_write(
                path=Path("config/settings.py"),
                before="# Configuration\nDEBUG = False",
                after="# Configuration\nDEBUG = True\nLOG_LEVEL = 'INFO'",
                note="enable debug"
            )

            # Verify both files were created
            patch_files = list(logger.file_dir.glob("*apply_patch.diff"))
            write_files = list(logger.file_dir.glob("*write_file.diff"))

            assert len(patch_files) == 1
            assert len(write_files) == 1

            # Verify content
            patch_content = patch_files[0].read_text(encoding='utf-8')
            assert patch_diff in patch_content

            write_content = write_files[0].read_text(encoding='utf-8')
            assert "--- a/config/settings.py" in write_content
            assert "+++ b/config/settings.py" in write_content
            assert "+LOG_LEVEL = 'INFO'" in write_content

    def test_logger_with_git_and_file_logging(self) -> None:
        """Test logger with both git and file logging enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_path, check=True, capture_output=True)

            logger = ChangeLogger(
                root=temp_path,
                enable_file_log=True,
                enable_git_commit=True,
                commit_prefix="TEST"
            )

            with patch("subprocess.run") as mock_subprocess:
                logger.log_patch("test patch", note="integration test")

                # Should create file and attempt git commit
                patch_files = list(logger.file_dir.glob("*"))
                assert len(patch_files) == 1

                # Should call git commands
                assert mock_subprocess.call_count >= 2

    def test_logger_handles_missing_git_directory(self) -> None:
        """Test logger handles missing .git directory gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Remove .git if it exists (shouldn't in fresh temp dir)
            git_dir = temp_path / ".git"
            if git_dir.exists():
                import shutil
                shutil.rmtree(git_dir)

            logger = ChangeLogger(root=temp_path, enable_git_commit=True)

            # Should not raise error when .git doesn't exist
            logger.log_patch("test", note="no git repo")


class TestChangeLoggerEdgeCases:
    """Test edge cases and error conditions."""

    def test_logger_with_special_characters_in_paths(self) -> None:
        """Test logger with special characters in file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            # Use path with special characters
            special_path = Path("test/file-with-spëcial-châractërs.py")

            logger.log_write(
                path=special_path,
                before="old content",
                after="new content"
            )

            # Should handle special characters
            files = list(logger.file_dir.glob("*"))
            assert len(files) == 1

            content = files[0].read_text(encoding='utf-8')
            assert "file-with-spëcial-châractërs.py" in content

    def test_logger_with_very_long_file_paths(self) -> None:
        """Test logger with very long file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            # Create very long path
            long_path = Path("very/long/path/to/file/with/a/very/long/name/that/exceeds/normal/lengths/main.py")

            logger.log_write(
                path=long_path,
                before="old",
                after="new"
            )

            # Should handle long paths
            files = list(logger.file_dir.glob("*"))
            assert len(files) == 1

    def test_concurrent_logging_operations(self) -> None:
        """Test handling concurrent logging operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            # Simulate concurrent logging
            def log_multiple_items(start_id: int, count: int) -> None:
                for i in range(count):
                    logger.log_patch(f"diff {start_id}_{i}", note=f"patch {start_id}_{i}")
                    logger.log_write(
                        path=Path(f"file_{start_id}_{i}.py"),
                        before=f"old {start_id}_{i}",
                        after=f"new {start_id}_{i}"
                    )

            # This is simplified - in real scenarios would need proper threading
            log_multiple_items(1, 5)
            log_multiple_items(2, 5)

            # Should have 20 files (10 patches + 10 writes)
            all_files = list(logger.file_dir.glob("*"))
            assert len(all_files) == 20

    def test_logger_with_binary_content(self) -> None:
        """Test logger with binary-like content in diffs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            # Diff with binary-like content (null bytes, special chars)
            binary_diff = """--- a/binary.dat
+++ b/binary.dat
@@ -1,5 +1,5 @@
-�PNG
+�PNG
 binary data
-\x00\x01\x02
+\x00\x01\x02\x03
"""

            logger.log_patch(binary_diff)

            # Should handle binary content
            files = list(logger.file_dir.glob("*"))
            assert len(files) == 1

            content = files[0].read_text(encoding='utf-8')
            assert binary_diff in content

    def test_logger_directory_creation_race_condition(self) -> None:
        """Test logger handles directory creation race conditions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create logger that needs to create nested directory
            nested_dir = temp_path / "deeply" / "nested" / "changes"
            logger = ChangeLogger(root=temp_path, file_dir=nested_dir)

            # Should handle exist_ok=True properly
            logger.log_patch("test diff")

            files = list(logger.file_dir.glob("*"))
            assert len(files) == 1

    def test_log_write_with_large_content(self) -> None:
        """Test logging file write with very large content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger = ChangeLogger(root=temp_path, enable_file_log=True)

            # Create large content
            large_before = "line\n" * 10000
            large_after = "modified line\n" * 10000

            logger.log_write(
                path=Path("large_file.py"),
                before=large_before,
                after=large_after
            )

            # Should handle large content
            files = list(logger.file_dir.glob("*"))
            assert len(files) == 1

            content = files[0].read_text(encoding='utf-8')
            # Should contain diff markers
            assert "--- a/large_file.py" in content
            assert "+++ b/large_file.py" in content