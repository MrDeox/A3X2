"""Code analysis engine for A3X execution system.

This module contains code analysis functionality including complexity analysis,
impact assessment, and quality metrics that was previously embedded in
the monolithic ActionExecutor class.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from typing import Dict, Any, List, Tuple

from ..actions import AgentAction
from ..cache import ast_cache_manager
from ..constants import MAX_DIFF_COMPLEXITY_SCORE, MAX_DIFF_LINES

class CodeAnalyzer:
    """Code analyzer for complexity, quality, and impact analysis.

    This class provides comprehensive code analysis capabilities including
    static analysis, complexity metrics, and impact assessment for changes.
    """

    def __init__(self, orchestrator) -> None:
        """Initialize the code analyzer.

        Args:
            orchestrator (ExecutionOrchestrator): The main execution orchestrator.
        """
        self.orchestrator = orchestrator

    def get_status(self) -> Dict[str, Any]:
        """Get status information about the code analyzer.

        Returns:
            Dict[str, Any]: Status information including analysis capabilities.
        """
        return {
            "analysis_types": [
                "impact_analysis", "complexity_analysis", "quality_analysis",
                "function_extraction", "class_extraction", "optimization_suggestions"
            ]
        }

    def analyze_impact_before_apply(self, action: AgentAction) -> Tuple[bool, str]:
        """Analyze the impact of a self-modification before applying it.

        Args:
            action (AgentAction): The self-modify action with diff to analyze.

        Returns:
            Tuple[bool, str]: (is_safe: bool, message: str)
        """
        if not action.diff:
            return False, "Empty diff for impact analysis"

        # Extract affected functions and classes
        affected_functions = self._extract_affected_functions(action.diff)
        affected_classes = self._extract_affected_classes(action.diff)

        # Check for dangerous changes
        if self._has_dangerous_self_change(action.diff):
            return False, "Dangerous change detected during impact analysis"

        # Check critical module modifications
        critical_modules = ["a3x/agent.py", "a3x/executor.py", "a3x/autoeval.py"]
        patch_paths = self._extract_paths_from_diff(action.diff)
        critical_changes = [p for p in patch_paths if any(cm in p for cm in critical_modules)]

        if critical_changes:
            if self._check_security_related_changes(action.diff):
                return False, f"Security-related changes detected in critical modules: {critical_changes}"

        # Analyze code quality
        quality_metrics = self._analyze_static_code_quality(action.diff)

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

        # Check complexity
        complexity_score = quality_metrics.get("complexity_score", 0)
        if complexity_score > MAX_DIFF_COMPLEXITY_SCORE:
            return False, f"Excessive complexity (score: {complexity_score})"

        # Check diff size
        diff_lines = len(action.diff.splitlines())
        if diff_lines > MAX_DIFF_LINES:
            return False, f"Diff too large for impact analysis ({diff_lines} lines)"

        # Check for test manipulation
        test_file_changes = [p for p in patch_paths if "test" in p.lower()]
        if test_file_changes and not any("test_autoeval" in p for p in test_file_changes):
            if self._check_test_manipulation(action.diff):
                return False, f"Suspicious test changes detected: {test_file_changes}"

        # Generate quality report
        quality_report = []
        if quality_metrics:
            quality_report.append(f"complexity: {quality_metrics.get('complexity_score', 0):.0f}")
            if quality_metrics.get("functions_added", 0) > 0:
                quality_report.append(f"new functions: {quality_metrics.get('functions_added', 0):.0f}")
            if quality_metrics.get("classes_added", 0) > 0:
                quality_report.append(f"new classes: {quality_metrics.get('classes_added', 0):.0f}")

        quality_msg = f" ({', '.join(quality_report)})" if quality_report else ""

        return True, f"Impact verified: {len(affected_functions)} functions, {len(affected_classes)} classes affected{quality_msg}"

    def _extract_affected_functions(self, diff: str) -> List[str]:
        """Extract functions that are being modified in the diff."""
        # Find added functions
        added_functions = re.findall(r"\+def\s+(\w+)", diff)
        # Find removed functions
        removed_functions = re.findall(r"-def\s+(\w+)", diff)
        modified_functions = list(set(added_functions + removed_functions))

        # Find functions in context
        all_context_lines = []
        lines = diff.split("\n")
        in_context = False
        current_function = None

        for line in lines:
            if line.startswith("@@"):
                in_context = True
            elif line.startswith("def ") and in_context:
                func_match = re.search(r"def\s+(\w+)", line)
                if func_match:
                    current_function = func_match.group(1)
                    all_context_lines.append(current_function)
            elif line.strip() == "" and current_function:
                current_function = None
                in_context = False

        # Get functions in affected contexts
        context_functions = re.findall(r"def\s+(\w+)", "\n".join(all_context_lines))
        modified_functions.extend(context_functions)

        return list(set(modified_functions))

    def _extract_affected_classes(self, diff: str) -> List[str]:
        """Extract classes that are being modified in the diff."""
        added_classes = re.findall(r"\+class\s+(\w+)", diff)
        removed_classes = re.findall(r"-class\s+(\w+)", diff)
        modified_classes = list(set(added_classes + removed_classes))

        return modified_classes

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

    def _check_test_manipulation(self, diff: str) -> bool:
        """Check if diff manipulates tests suspiciously."""
        # Count assertion changes
        removed_assertions = diff.count("-assert") + diff.count("-self.assertTrue") + diff.count("-self.assertFalse")
        added_assertions = diff.count("+assert") + diff.count("+self.assertTrue") + diff.count("+self.assertFalse")

        # If removing more assertions than adding, might be manipulation
        return removed_assertions > added_assertions

    def _analyze_static_code_quality(self, diff: str) -> Dict[str, float]:
        """Analyze static code quality in the diff."""
        quality_metrics = {}

        # Extract Python code from diff
        python_code = self._extract_python_code_from_diff(diff)
        if not python_code:
            return quality_metrics

        # Analyze complexity with caching
        ast_cache = ast_cache_manager.get_cache("analysis_complexity")
        try:
            tree, syntax_valid = ast_cache.get_parse_result(python_code)
            if syntax_valid:
                complexity_stats = self._analyze_code_complexity(tree)
            else:
                quality_metrics["syntax_errors"] = 1.0
                return quality_metrics
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
        bad_practices = self._check_bad_coding_practices(python_code)
        quality_metrics.update(bad_practices)

        return quality_metrics

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

    def calculate_cyclomatic_complexity(self, code: str) -> Dict[str, float]:
        """Calculate cyclomatic complexity of code.

        Args:
            code (str): The code to analyze.

        Returns:
            Dict[str, float]: Complexity metrics.
        """
        ast_cache = ast_cache_manager.get_cache("complexity_analysis")
        try:
            tree, syntax_valid = ast_cache.get_parse_result(code)
            if not syntax_valid:
                return {"syntax_error": 1.0}
        except SyntaxError:
            return {"syntax_error": 1.0}

        complexity_metrics = {
            "total_complexity": 1.0,  # Base complexity
            "function_count": 0.0,
            "average_function_complexity": 0.0,
            "max_function_complexity": 0.0,
            "decision_points": 0.0,
        }

        function_complexities = []

        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.decision_points = 0
                self.function_stack = []

            def visit_FunctionDef(self, node):
                self.function_stack.append({"name": node.name, "complexity": 1, "nodes": []})
                self.generic_visit(node)
                if self.function_stack:
                    func_info = self.function_stack.pop()
                    function_complexities.append(func_info["complexity"])

            def _increment_complexity(self, node):
                if self.function_stack:
                    self.function_stack[-1]["complexity"] += 1
                    self.function_stack[-1]["nodes"].append(node)
                self.decision_points += 1

            def visit_If(self, node):
                self._increment_complexity(node)
                self.generic_visit(node)

            def visit_For(self, node):
                self._increment_complexity(node)
                self.generic_visit(node)

            def visit_While(self, node):
                self._increment_complexity(node)
                self.generic_visit(node)

            def visit_Try(self, node):
                self._increment_complexity(node)
                self.generic_visit(node)

        visitor = ComplexityVisitor()
        visitor.visit(tree)

        complexity_metrics["decision_points"] = float(visitor.decision_points)
        complexity_metrics["function_count"] = float(len(function_complexities))

        if function_complexities:
            complexity_metrics["average_function_complexity"] = float(sum(function_complexities) / len(function_complexities))
            complexity_metrics["max_function_complexity"] = float(max(function_complexities))
            complexity_metrics["total_complexity"] = float(1 + visitor.decision_points)

        return complexity_metrics

    def generate_optimization_suggestions(self, code: str, quality_metrics: Dict[str, float]) -> List[str]:
        """Generate optimization suggestions based on code analysis."""
        suggestions = []

        if quality_metrics.get("magic_numbers", 0) > 0:
            suggestions.append("Replace magic numbers with named constants")

        if quality_metrics.get("global_vars", 0) > 0:
            suggestions.append("Convert global variables to parameters or class attributes")

        if quality_metrics.get("hardcoded_paths", 0) > 0:
            suggestions.append("Use configuration or environment variables for paths")

        if quality_metrics.get("complexity_score", 0) > 100:
            suggestions.append("Consider splitting function into smaller parts")

        if quality_metrics.get("max_nesting_depth", 0) > 5:
            suggestions.append("Reduce nesting using guard clauses or method extraction")

        # Analyze for specific patterns
        self._analyze_code_for_specific_suggestions(code, suggestions, quality_metrics)

        return suggestions

    def _analyze_code_for_specific_suggestions(self, code: str, suggestions: List[str], quality_metrics: Dict[str, float]) -> None:
        """Analyze code for specific optimization opportunities."""
        # Check for loops that could be optimized
        for_loop_matches = re.findall(r"for\s+\w+\s+in\s+range\(", code)
        if for_loop_matches:
            suggestions.append("Consider using list comprehensions or built-in functions for simple loops")

        # Check for string concatenation in loops
        if "+=" in code and ("for" in code or "while" in code):
            suggestions.append("Use ''.join() for string concatenation in loops")

        # Check for excessive lambda usage
        lambda_usage = re.findall(r"lambda\s+.*:.*[^,\\)]", code)
        if len(lambda_usage) > 2:
            suggestions.append("Consider replacing lambdas with named functions for better readability")

        # Check for unused imports
        unused_imports = self._check_unused_imports(code)
        if unused_imports:
            suggestions.append(f"Remove unused imports: {', '.join(unused_imports)}")

        # Check for unused variables
        unused_vars = self._check_unused_variables(code)
        if unused_vars:
            suggestions.append(f"Remove unused variables: {', '.join(unused_vars)}")

    def _check_unused_imports(self, code: str) -> List[str]:
        """Check for unused imports."""
        ast_cache = ast_cache_manager.get_cache("unused_imports")
        try:
            tree, syntax_valid = ast_cache.get_parse_result(code)
            if not syntax_valid:
                return []
        except SyntaxError:
            return []

        # Find all imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split(".")[0])

        # Find used names
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)

        # Check for unused imports
        unused = []
        for imp in imports:
            if imp not in used_names and imp not in ["typing", "os", "sys"]:
                unused.append(imp)

        return unused

    def _check_unused_variables(self, code: str) -> List[str]:
        """Check for unused variables."""
        ast_cache = ast_cache_manager.get_cache("unused_variables")
        try:
            tree, syntax_valid = ast_cache.get_parse_result(code)
            if not syntax_valid:
                return []
        except SyntaxError:
            return []

        assigned_vars = set()
        used_vars = set()

        class VariableVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
                self.generic_visit(node)

            def visit_AugAssign(self, node):
                if isinstance(node.target, ast.Name):
                    assigned_vars.add(node.target.id)
                self.generic_visit(node)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
                self.generic_visit(node)

        visitor = VariableVisitor()
        visitor.visit(tree)

        unused = list(assigned_vars - used_vars)
        return [var for var in unused if not var.startswith("_")]