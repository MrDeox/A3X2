"""Validation engine for A3X execution system.

This module contains all validation logic including pre/post execution
validation, syntax checking, and risk assessment that was previously
embedded in the monolithic ActionExecutor class.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Dict, Any, List, Tuple

from ..actions import AgentAction, Observation
from ..patch import PatchManager

class ValidationEngine:
    """Validation engine for pre and post-execution checks.

    This class handles all validation logic including syntax checking,
    risk assessment, and policy validation for actions before and after
    execution.
    """

    def __init__(self, orchestrator) -> None:
        """Initialize the validation engine.

        Args:
            orchestrator (ExecutionOrchestrator): The main execution orchestrator.
        """
        self.orchestrator = orchestrator
        self.workspace_root = orchestrator.workspace_root
        self.patch_manager = orchestrator.patch_manager

    def get_status(self) -> Dict[str, Any]:
        """Get status information about the validation engine.

        Returns:
            Dict[str, Any]: Status information including validation capabilities.
        """
        return {
            "validation_types": [
                "pre_execution", "post_execution", "syntax",
                "risk_assessment", "policy", "impact_analysis"
            ],
            "workspace_root": str(self.workspace_root),
        }

    def validate_pre_execution(self, action: AgentAction) -> Observation:
        """Perform pre-execution validation on an action.

        Args:
            action (AgentAction): The action to validate.

        Returns:
            Observation: Validation result - success if valid, error if invalid.
        """
        # Basic validation
        if not action.type:
            return Observation(
                success=False,
                error="Action type not specified",
                type="validation"
            )

        # Path validation for file operations
        if action.type.name in ["READ_FILE", "WRITE_FILE"] and not action.path:
            return Observation(
                success=False,
                error=f"Path required for {action.type.name} actions",
                type="validation"
            )

        # Content validation for write operations
        if action.type.name == "WRITE_FILE" and action.content is None:
            return Observation(
                success=False,
                error="Content required for WRITE_FILE actions",
                type="validation"
            )

        # Command validation for run operations
        if action.type.name == "RUN_COMMAND" and not action.command:
            return Observation(
                success=False,
                error="Command required for RUN_COMMAND actions",
                type="validation"
            )

        # Diff validation for patch operations
        if action.type.name in ["APPLY_PATCH", "SELF_MODIFY"] and not action.diff:
            return Observation(
                success=False,
                error=f"Diff required for {action.type.name} actions",
                type="validation"
            )

        # Policy validation
        policy_result = self._validate_policy(action)
        if not policy_result.success:
            return policy_result

        return Observation(success=True, output="Pre-execution validation passed", type="validation")

    def validate_post_execution(self, action: AgentAction, result: Observation) -> None:
        """Perform post-execution validation and logging.

        Args:
            action (AgentAction): The action that was executed.
            result (Observation): The result of the execution.
        """
        # Log execution results
        if not result.success:
            logging.warning(f"Action {action.type.name} failed: {result.error}")

        # Additional post-execution checks could be added here
        # For example, checking file integrity after modifications

    def _validate_policy(self, action: AgentAction) -> Observation:
        """Validate action against security policies.

        Args:
            action (AgentAction): The action to validate.

        Returns:
            Observation: Validation result.
        """
        # Self-modify restrictions
        if action.type.name == "SELF_MODIFY":
            if action.diff:
                allowed_prefixes = ["a3x", "configs"]
                patch_paths = self.patch_manager.extract_paths(action.diff)
                invalid_paths = [p for p in patch_paths if not any(p.startswith(prefix) for prefix in allowed_prefixes)]

                if invalid_paths:
                    return Observation(
                        success=False,
                        error=f"Self-modify restricted to a3x/ and configs/: invalid {invalid_paths}",
                        type="policy_validation"
                    )

        # Command restrictions
        if action.type.name == "RUN_COMMAND":
            if action.command and not self._command_allowed_by_policy(action.command):
                return Observation(
                    success=False,
                    error="Command blocked by security policy",
                    type="policy_validation"
                )

        return Observation(success=True, output="Policy validation passed", type="policy_validation")

    def _command_allowed_by_policy(self, command: list[str]) -> bool:
        """Check if command is allowed by current policies."""
        joined = " ".join(command)
        for pattern in self.orchestrator.config.policies.deny_commands:
            if pattern in joined:
                return False
        return True

    def validate_patch_syntax(self, py_paths: set[str], original_states: dict, backups: dict) -> Tuple[bool, str]:
        """Validate patch for syntax errors and rollback on failure.

        Args:
            py_paths: Set of Python file paths affected by the patch.
            original_states: Original content of files for rollback.
            backups: Backup file paths for cleanup.

        Returns:
            Tuple[bool, str]: (is_valid, error_output)
        """
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

    def validate_self_modify_safety(self, action: AgentAction) -> Tuple[bool, str]:
        """Validate self-modification for dangerous changes.

        Args:
            action (AgentAction): The self-modify action to validate.

        Returns:
            Tuple[bool, str]: (is_safe, message)
        """
        if not action.diff:
            return False, "Empty diff for self-modify validation"

        # Check for dangerous patterns
        if self._has_dangerous_self_change(action.diff):
            return False, "Dangerous change detected during self-modify validation"

        # Check critical module modifications
        critical_modules = ["a3x/agent.py", "a3x/executor.py", "a3x/autoeval.py"]
        patch_paths = self._extract_paths_from_diff(action.diff)
        critical_changes = [p for p in patch_paths if any(cm in p for cm in critical_modules)]

        if critical_changes:
            if self._check_security_related_changes(action.diff):
                return False, f"Security-related changes detected in critical modules: {critical_changes}"

        return True, "Self-modify validation passed"

    def _has_dangerous_self_change(self, diff: str) -> bool:
        """Check for dangerous patterns in self-modification diff."""
        dangerous_patterns = [
            r"\+.*allow_network.*=.*True",  # Enabling network
            r"-.*deny_commands.*=\[\]",     # Clearing denials
            r"\+.*sudo",                    # Adding privileges
            r"-.*_is_safe_command",         # Removing safety checks
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, diff):
                return True
        return False

    def _extract_paths_from_diff(self, diff: str) -> List[str]:
        """Extract file paths from a diff."""
        old_file_pattern = r"^--- a/(.+)$"
        new_file_pattern = r"^\+\+\+ b/(.+)$"

        paths = set()
        lines_diff = diff.split("\n")

        for line in lines_diff:
            old_match = re.match(old_file_pattern, line.strip())
            if old_match:
                paths.add(old_match.group(1))

            new_match = re.match(new_file_pattern, line.strip())
            if new_match:
                paths.add(new_match.group(1))

        return list(paths)

    def _check_security_related_changes(self, diff: str) -> bool:
        """Check if diff contains security-related changes."""
        security_keywords = [
            "allow_network", "deny_commands", "is_safe_command", "command_allowed",
            "safe", "security", "permission", "privilege", "admin", "root", "sudo"
        ]
        diff_lower = diff.lower()
        return any(keyword in diff_lower for keyword in security_keywords)

    def validate_diff_complexity(self, diff: str) -> Tuple[bool, str]:
        """Validate that diff complexity is within acceptable limits.

        Args:
            diff (str): The diff to validate.

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        from ..constants import MAX_DIFF_LINES

        diff_lines = len(diff.splitlines())
        if diff_lines > MAX_DIFF_LINES:
            return False, f"Diff too large for validation ({diff_lines} lines), max allowed: {MAX_DIFF_LINES}"

        return True, f"Diff complexity validated ({diff_lines} lines)"

    def validate_code_quality(self, diff: str) -> Tuple[bool, str]:
        """Validate code quality metrics in the diff.

        Args:
            diff (str): The diff to validate.

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        # Extract Python code from diff
        python_code = self._extract_python_code_from_diff(diff)
        if not python_code:
            return True, "No Python code to validate"

        # Analyze quality metrics
        quality_metrics = self._analyze_static_code_quality(python_code)

        # Check for quality issues
        quality_issues = []
        if quality_metrics.get("syntax_errors", 0) > 0:
            quality_issues.append("syntax errors")
        if quality_metrics.get("magic_numbers", 0) > 5:
            quality_issues.append("magic numbers")
        if quality_metrics.get("global_vars", 0) > 2:
            quality_issues.append("excessive global variables")
        if quality_metrics.get("long_functions", 0) > 0:
            quality_issues.append("long functions")

        if quality_issues:
            return False, f"Code quality issues: {', '.join(quality_issues)}"

        return True, "Code quality validation passed"

    def _extract_python_code_from_diff(self, diff: str) -> str:
        """Extract Python code from a diff."""
        lines = diff.split("\n")
        python_code = []

        in_diff = False
        for line in lines:
            if line.startswith("+++ ") and line.endswith(".py"):
                in_diff = True
                continue
            elif line.startswith("--- ") or line.startswith("@@ "):
                continue
            elif line.startswith(" ") or line.startswith("+"):
                if in_diff:
                    code_line = line[1:]  # Remove the prefix
                    python_code.append(code_line)

        return "\n".join(python_code)

    def _analyze_static_code_quality(self, code: str) -> Dict[str, float]:
        """Analyze static code quality metrics."""
        quality_metrics = {}

        try:
            import ast
            tree = ast.parse(code)
            complexity_stats = self._analyze_code_complexity(tree)
            quality_metrics.update({
                "functions_added": float(complexity_stats["function_count"]),
                "classes_added": float(complexity_stats["class_count"]),
                "complexity_score": float(complexity_stats["total_nodes"]),
                "max_nesting_depth": float(complexity_stats["max_depth"])
            })
        except SyntaxError:
            quality_metrics["syntax_errors"] = 1.0
            return quality_metrics

        # Check for bad practices
        bad_practices = self._check_bad_coding_practices(code)
        quality_metrics.update(bad_practices)

        return quality_metrics

    def _analyze_code_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Analyze code complexity from AST."""
        stats = {
            "function_count": 0,
            "class_count": 0,
            "total_nodes": 0,
            "max_depth": 0,
        }

        def count_nodes(node, depth=0):
            stats["total_nodes"] += 1
            stats["max_depth"] = max(stats["max_depth"], depth)

            if isinstance(node, ast.FunctionDef):
                stats["function_count"] += 1
            elif isinstance(node, ast.ClassDef):
                stats["class_count"] += 1

            for child in ast.iter_child_nodes(node):
                count_nodes(child, depth + 1)

        for node in ast.iter_child_nodes(tree):
            count_nodes(node)

        return stats

    def _check_bad_coding_practices(self, code: str) -> Dict[str, float]:
        """Check for bad coding practices."""
        bad_practices = {}

        # Global variables
        if "global " in code:
            bad_practices["global_vars"] = float(code.count("global "))

        # Magic numbers
        magic_numbers = len(re.findall(r"[^a-zA-Z_]\d+\.?\d*", code))
        if magic_numbers > 5:
            bad_practices["magic_numbers"] = float(magic_numbers)

        # Hardcoded paths/URLs
        hardcoded_paths = len(re.findall(r'[\'"].*[/\\].*[\'"]', code))
        if hardcoded_paths > 0:
            bad_practices["hardcoded_paths"] = float(hardcoded_paths)

        # Long functions (>50 lines)
        lines = code.split("\n")
        long_functions = 0
        current_function_lines = 0
        in_function = False

        for line in lines:
            if line.strip().startswith("def "):
                if in_function and current_function_lines > 50:
                    long_functions += 1
                in_function = True
                current_function_lines = 1
            elif in_function:
                if line.strip() == "":
                    continue
                current_function_lines += 1

        if in_function and current_function_lines > 50:
            long_functions += 1

        if long_functions > 0:
            bad_practices["long_functions"] = float(long_functions)

        return bad_practices