"""Configuration testing utilities for validation and testing."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from a3x.config import load_config, AgentConfig
from a3x.config.validation import validate_config_file, ValidationError


@dataclass
class TestResult:
    """Result of a configuration test."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        output = f"[{status}] {self.test_name}: {self.message}"
        if self.duration is not None:
            output += f" ({self.duration:.3f}s)"
        return output


class ConfigTester:
    """Configuration testing and validation utilities."""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.test_results: List[TestResult] = []

    def run_validation_tests(self) -> List[TestResult]:
        """Run all validation tests on the configuration."""
        results = []

        # Test 1: Schema validation
        results.append(self._test_schema_validation())

        # Test 2: File existence tests
        results.append(self._test_file_existence())

        # Test 3: Configuration loading test
        results.append(self._test_config_loading())

        # Test 4: Value range tests
        results.append(self._test_value_ranges())

        # Test 5: Dependency tests
        results.append(self._test_dependencies())

        self.test_results.extend(results)
        return results

    def run_functional_tests(self) -> List[TestResult]:
        """Run functional tests that actually use the configuration."""
        results = []

        # Test 6: Command execution tests (if tests are configured)
        results.append(self._test_command_execution())

        # Test 7: Environment variable tests
        results.append(self._test_environment_variables())

        self.test_results.extend(results)
        return results

    def run_all_tests(self) -> List[TestResult]:
        """Run all available tests."""
        all_results = []
        all_results.extend(self.run_validation_tests())
        all_results.extend(self.run_functional_tests())
        return all_results

    def _test_schema_validation(self) -> TestResult:
        """Test configuration against JSON schema."""
        try:
            validate_config_file(self.config_path, strict=True)
            return TestResult(
                "Schema Validation",
                True,
                "Configuration matches JSON schema"
            )
        except ValidationError as e:
            return TestResult(
                "Schema Validation",
                False,
                str(e),
                {"errors": e.errors}
            )
        except ImportError:
            return TestResult(
                "Schema Validation",
                False,
                "JSON schema validation not available (missing jsonschema package)"
            )
        except Exception as e:
            return TestResult(
                "Schema Validation",
                False,
                f"Unexpected error: {e}"
            )

    def _test_file_existence(self) -> TestResult:
        """Test that referenced files exist."""
        try:
            config_data = self._load_config_data()
            issues = []

            # Check workspace root
            if "workspace" in config_data:
                workspace_root = config_data["workspace"].get("root", ".")
                if isinstance(workspace_root, str):
                    full_path = self._resolve_path(workspace_root)
                    if not full_path.exists():
                        issues.append(f"Workspace root does not exist: {full_path}")

            # Check seed backlog
            if "loop" in config_data:
                seed_backlog = config_data["loop"].get("seed_backlog")
                if seed_backlog:
                    full_path = self._resolve_path(seed_backlog)
                    if not full_path.exists():
                        issues.append(f"Seed backlog does not exist: {full_path}")

            if issues:
                return TestResult(
                    "File Existence",
                    False,
                    "; ".join(issues)
                )
            else:
                return TestResult(
                    "File Existence",
                    True,
                    "All referenced files exist"
                )

        except Exception as e:
            return TestResult(
                "File Existence",
                False,
                f"Error checking file existence: {e}"
            )

    def _test_config_loading(self) -> TestResult:
        """Test that configuration can be loaded successfully."""
        try:
            config = load_config(self.config_path, validate=False)  # Don't double-validate
            return TestResult(
                "Configuration Loading",
                True,
                f"Successfully loaded configuration for workspace: {config.workspace_root}"
            )
        except Exception as e:
            return TestResult(
                "Configuration Loading",
                False,
                f"Failed to load configuration: {e}"
            )

    def _test_value_ranges(self) -> TestResult:
        """Test that configuration values are within acceptable ranges."""
        try:
            config_data = self._load_config_data()
            issues = []

            # Check limits
            if "limits" in config_data:
                limits = config_data["limits"]

                for field, (min_val, max_val) in {
                    "max_iterations": (1, 1000),
                    "command_timeout": (1, 3600),
                    "max_failures": (1, 100)
                }.items():
                    if field in limits:
                        value = limits[field]
                        if not isinstance(value, int) or value < min_val or value > max_val:
                            issues.append(f"{field} must be between {min_val} and {max_val}, got: {value}")

            # Check goals (should be between 0.0 and 1.0)
            if "goals" in config_data:
                goals = config_data["goals"]
                for goal_name, goal_value in goals.items():
                    if isinstance(goal_value, dict) and "min" in goal_value:
                        value = goal_value["min"]
                    elif isinstance(goal_value, (int, float)):
                        value = goal_value
                    else:
                        continue

                    if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
                        issues.append(f"Goal '{goal_name}' must be between 0.0 and 1.0, got: {value}")

            if issues:
                return TestResult(
                    "Value Ranges",
                    False,
                    "; ".join(issues)
                )
            else:
                return TestResult(
                    "Value Ranges",
                    True,
                    "All values are within acceptable ranges"
                )

        except Exception as e:
            return TestResult(
                "Value Ranges",
                False,
                f"Error checking value ranges: {e}"
            )

    def _test_dependencies(self) -> TestResult:
        """Test that configuration dependencies are satisfied."""
        try:
            config_data = self._load_config_data()
            issues = []

            # Check LLM configuration
            if "llm" in config_data:
                llm_config = config_data["llm"]

                # Non-manual LLM types need model
                if llm_config.get("type") != "manual" and "model" not in llm_config:
                    issues.append("LLM model is required for non-manual LLM types")

                # API-based LLMs need API key environment variable
                api_types = ["openai", "anthropic", "openrouter"]
                if llm_config.get("type") in api_types and "api_key_env" not in llm_config:
                    issues.append(f"LLM type '{llm_config['type']}' requires api_key_env")

            if issues:
                return TestResult(
                    "Dependencies",
                    False,
                    "; ".join(issues)
                )
            else:
                return TestResult(
                    "Dependencies",
                    True,
                    "All dependencies are satisfied"
                )

        except Exception as e:
            return TestResult(
                "Dependencies",
                False,
                f"Error checking dependencies: {e}"
            )

    def _test_command_execution(self) -> TestResult:
        """Test that configured test commands can be executed."""
        try:
            config = load_config(self.config_path, validate=False)

            if not config.tests.auto or not config.tests.commands:
                return TestResult(
                    "Command Execution",
                    True,
                    "No test commands configured or auto-testing disabled"
                )

            successful_commands = 0
            total_commands = len(config.tests.commands)

            for cmd in config.tests.commands:
                try:
                    # Try to execute command (with timeout and dry-run if possible)
                    if isinstance(cmd, list):
                        cmd_args = cmd
                    else:
                        cmd_args = cmd.split()

                    # Just check if command exists in PATH, don't actually run it
                    result = subprocess.run(
                        ["which", cmd_args[0]],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode == 0:
                        successful_commands += 1

                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                    continue

            if successful_commands == total_commands:
                return TestResult(
                    "Command Execution",
                    True,
                    f"All {total_commands} test commands are available"
                )
            elif successful_commands > 0:
                return TestResult(
                    "Command Execution",
                    False,
                    f"{successful_commands}/{total_commands} test commands are available"
                )
            else:
                return TestResult(
                    "Command Execution",
                    False,
                    "No test commands are available in PATH"
                )

        except Exception as e:
            return TestResult(
                "Command Execution",
                False,
                f"Error testing command execution: {e}"
            )

    def _test_environment_variables(self) -> TestResult:
        """Test that required environment variables are available."""
        try:
            config_data = self._load_config_data()
            issues = []

            # Check for API key environment variables
            if "llm" in config_data:
                llm_config = config_data["llm"]
                api_key_env = llm_config.get("api_key_env")

                if api_key_env:
                    import os
                    if api_key_env not in os.environ:
                        issues.append(f"Required environment variable '{api_key_env}' is not set")

            if issues:
                return TestResult(
                    "Environment Variables",
                    False,
                    "; ".join(issues)
                )
            else:
                return TestResult(
                    "Environment Variables",
                    True,
                    "All required environment variables are set"
                )

        except Exception as e:
            return TestResult(
                "Environment Variables",
                False,
                f"Error checking environment variables: {e}"
            )

    def _load_config_data(self) -> Dict[str, Any]:
        """Load raw configuration data."""
        import yaml
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the configuration file."""
        if Path(path).is_absolute():
            return Path(path).resolve()
        else:
            return (self.config_path.parent / path).resolve()

    def print_results(self, results: Optional[List[TestResult]] = None) -> None:
        """Print test results to console."""
        if results is None:
            results = self.test_results

        if not results:
            print("No tests have been run.")
            return

        print(f"\nConfiguration Test Results for: {self.config_path}")
        print("=" * 60)

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        for result in results:
            print(result)

        print("-" * 60)
        print(f"Summary: {passed}/{total} tests passed")

        if passed < total:
            print(f"\nFailed tests: {total - passed}")
            for result in results:
                if not result.passed:
                    print(f"  - {result.test_name}")
        else:
            print("All tests passed! ✓")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all test results."""
        if not self.test_results:
            return {"total": 0, "passed": 0, "failed": 0, "results": []}

        passed = sum(1 for r in self.test_results if r.passed)
        failed = len(self.test_results) - passed

        return {
            "total": len(self.test_results),
            "passed": passed,
            "failed": failed,
            "results": [r.__dict__ for r in self.test_results]
        }