"""Configuration analysis tools for troubleshooting and optimization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from a3x.config import load_config, AgentConfig
from a3x.config.validation.schemas import CONFIG_SCHEMA


class ConfigAnalyzer:
    """Analyzes configuration files for issues, optimizations, and best practices."""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.issues: List[Dict[str, Any]] = []
        self.optimizations: List[Dict[str, Any]] = []
        self.security_concerns: List[Dict[str, Any]] = []

    def analyze_comprehensive(self) -> Dict[str, Any]:
        """Perform comprehensive configuration analysis."""
        self.issues.clear()
        self.optimizations.clear()
        self.security_concerns.clear()

        # Load configuration data
        config_data = self._load_config_data()

        # Run all analysis types
        self._analyze_security()
        self._analyze_performance()
        self._analyze_maintainability()
        self._analyze_best_practices()
        self._analyze_dependencies()

        return {
            "config_path": str(self.config_path),
            "issues": self.issues,
            "optimizations": self.optimizations,
            "security_concerns": self.security_concerns,
            "summary": self._generate_summary()
        }

    def _analyze_security(self) -> None:
        """Analyze configuration for security issues."""
        config_data = self._load_config_data()

        # Check for overly permissive settings
        if "policies" in config_data:
            policies = config_data["policies"]

            if policies.get("allow_network", False):
                self.security_concerns.append({
                    "type": "warning",
                    "category": "security",
                    "message": "Network access is enabled, ensure this is intentional",
                    "field": "policies.allow_network",
                    "current_value": True,
                    "recommendation": "Disable if network access is not required"
                })

            if policies.get("allow_shell_write", False):
                self.security_concerns.append({
                    "type": "info",
                    "category": "security",
                    "message": "Shell write operations are allowed",
                    "field": "policies.allow_shell_write",
                    "current_value": True,
                    "recommendation": "Verify this is required for your use case"
                })

            # Check for dangerous commands in deny list
            deny_commands = policies.get("deny_commands", [])
            dangerous_commands = ["rm -rf", "sudo", "su", "chmod 777", "chown"]
            for cmd in dangerous_commands:
                if cmd not in deny_commands:
                    self.optimizations.append({
                        "type": "enhancement",
                        "category": "security",
                        "message": f"Consider adding '{cmd}' to deny_commands",
                        "field": "policies.deny_commands",
                        "recommendation": f"Add '{cmd}' to the deny list for better security"
                    })

    def _analyze_performance(self) -> None:
        """Analyze configuration for performance issues and optimizations."""
        config_data = self._load_config_data()

        # Check limits for performance impact
        if "limits" in config_data:
            limits = config_data["limits"]

            max_iterations = limits.get("max_iterations", 50)
            if max_iterations > 100:
                self.optimizations.append({
                    "type": "warning",
                    "category": "performance",
                    "message": f"High max_iterations ({max_iterations}) may impact performance",
                    "field": "limits.max_iterations",
                    "current_value": max_iterations,
                    "recommendation": "Consider reducing max_iterations for better performance"
                })

            command_timeout = limits.get("command_timeout", 120)
            if command_timeout > 300:
                self.optimizations.append({
                    "type": "info",
                    "category": "performance",
                    "message": f"Long command timeout ({command_timeout}s) may block execution",
                    "field": "limits.command_timeout",
                    "current_value": command_timeout,
                    "recommendation": "Consider reducing timeout for faster failure detection"
                })

        # Check memory settings
        if "loop" in config_data:
            loop = config_data["loop"]

            memory_top_k = loop.get("memory_top_k", 3)
            if memory_top_k > 10:
                self.optimizations.append({
                    "type": "info",
                    "category": "performance",
                    "message": f"High memory_top_k ({memory_top_k}) may impact performance",
                    "field": "loop.memory_top_k",
                    "current_value": memory_top_k,
                    "recommendation": "Consider reducing memory_top_k for better performance"
                })

    def _analyze_maintainability(self) -> None:
        """Analyze configuration for maintainability issues."""
        config_data = self._load_config_data()

        # Check for absolute paths that may cause portability issues
        path_fields = [
            ("workspace.root", "workspace", "root"),
            ("loop.seed_backlog", "loop", "seed_backlog"),
            ("loop.seed_config", "loop", "seed_config"),
            ("audit.file_dir", "audit", "file_dir"),
        ]

        for field_path, section, field in path_fields:
            if section in config_data and field in config_data[section]:
                path_value = config_data[section][field]
                if isinstance(path_value, str) and Path(path_value).is_absolute():
                    self.optimizations.append({
                        "type": "info",
                        "category": "maintainability",
                        "message": f"Absolute path in {field_path} may reduce portability",
                        "field": field_path,
                        "current_value": path_value,
                        "recommendation": "Consider using relative paths for better portability"
                    })

        # Check for missing documentation/comments
        required_sections = ["llm", "workspace", "policies"]
        for section in required_sections:
            if section not in config_data:
                self.issues.append({
                    "type": "error",
                    "category": "maintainability",
                    "message": f"Required section '{section}' is missing",
                    "field": section,
                    "recommendation": f"Add the '{section}' section to your configuration"
                })

    def _analyze_best_practices(self) -> None:
        """Analyze configuration against best practices."""
        config_data = self._load_config_data()

        # Check for reasonable goal thresholds
        if "goals" in config_data:
            goals = config_data["goals"]

            for goal_name, goal_value in goals.items():
                if isinstance(goal_value, dict) and "min" in goal_value:
                    threshold = goal_value["min"]
                elif isinstance(goal_value, (int, float)):
                    threshold = goal_value
                else:
                    continue

                if threshold > 0.95:
                    self.optimizations.append({
                        "type": "info",
                        "category": "best_practice",
                        "message": f"Very high threshold ({threshold}) for {goal_name}",
                        "field": f"goals.{goal_name}",
                        "current_value": threshold,
                        "recommendation": "Consider if such high thresholds are realistic"
                    })
                elif threshold < 0.5:
                    self.optimizations.append({
                        "type": "info",
                        "category": "best_practice",
                        "message": f"Very low threshold ({threshold}) for {goal_name}",
                        "field": f"goals.{goal_name}",
                        "current_value": threshold,
                        "recommendation": "Consider if such low thresholds are appropriate"
                    })

        # Check for test configuration best practices
        if "tests" in config_data:
            tests = config_data["tests"]

            if tests.get("auto", False) and not tests.get("commands"):
                self.optimizations.append({
                    "type": "warning",
                    "category": "best_practice",
                    "message": "Auto-testing enabled but no test commands configured",
                    "field": "tests.commands",
                    "recommendation": "Add test commands or disable auto-testing"
                })

    def _analyze_dependencies(self) -> None:
        """Analyze configuration dependencies and requirements."""
        config_data = self._load_config_data()

        # Check LLM configuration completeness
        if "llm" in config_data:
            llm = config_data["llm"]
            llm_type = llm.get("type")

            if llm_type == "openrouter":
                missing_fields = []
                for field in ["model", "api_key_env"]:
                    if field not in llm:
                        missing_fields.append(field)

                if missing_fields:
                    self.issues.append({
                        "type": "error",
                        "category": "dependency",
                        "message": f"Missing required fields for OpenRouter: {missing_fields}",
                        "field": "llm",
                        "recommendation": "Add the missing fields to your LLM configuration"
                    })

            elif llm_type in ["openai", "anthropic"]:
                missing_fields = []
                for field in ["model", "api_key_env"]:
                    if field not in llm:
                        missing_fields.append(field)

                if missing_fields:
                    self.issues.append({
                        "type": "error",
                        "category": "dependency",
                        "message": f"Missing required fields for {llm_type}: {missing_fields}",
                        "field": "llm",
                        "recommendation": "Add the missing fields to your LLM configuration"
                    })

    def _load_config_data(self) -> Dict[str, Any]:
        """Load raw configuration data."""
        import yaml
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate analysis summary."""
        total_issues = len([i for i in self.issues if i["type"] == "error"])
        total_warnings = len([i for i in self.issues if i["type"] == "warning"]) + len(self.optimizations)
        total_info = len([i for i in self.issues if i["type"] == "info"])

        return {
            "total_issues": len(self.issues),
            "total_optimizations": len(self.optimizations),
            "total_security_concerns": len(self.security_concerns),
            "error_count": total_issues,
            "warning_count": total_warnings,
            "info_count": total_info,
            "overall_score": self._calculate_score()
        }

    def _calculate_score(self) -> int:
        """Calculate overall configuration quality score (0-100)."""
        score = 100

        # Deduct points for issues
        for issue in self.issues:
            if issue["type"] == "error":
                score -= 20
            elif issue["type"] == "warning":
                score -= 5

        # Deduct points for security concerns
        for concern in self.security_concerns:
            if concern["type"] == "warning":
                score -= 10
            elif concern["type"] == "info":
                score -= 2

        return max(0, min(100, score))

    def print_analysis(self, analysis_result: Optional[Dict[str, Any]] = None) -> None:
        """Print analysis results in a formatted way."""
        if analysis_result is None:
            analysis_result = self.analyze_comprehensive()

        print("\nConfiguration Analysis Report")
        print(f"File: {analysis_result['config_path']}")
        print("=" * 60)

        summary = analysis_result["summary"]

        # Overall score
        score = summary["overall_score"]
        score_color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        print(f"Overall Score: {score_color} {score}/100")

        # Summary counts
        print("\nSummary:")
        print(f"  Errors: {summary['error_count']}")
        print(f"  Warnings: {summary['warning_count']}")
        print(f"  Info: {summary['info_count']}")
        print(f"  Optimizations: {summary['total_optimizations']}")
        print(f"  Security Concerns: {summary['total_security_concerns']}")

        # Issues
        if analysis_result["issues"]:
            print("\n❌ Issues:")
            for issue in analysis_result["issues"]:
                icon = "🚨" if issue["type"] == "error" else "⚠️" if issue["type"] == "warning" else "ℹ️"
                print(f"  {icon} {issue['message']}")
                if "recommendation" in issue:
                    print(f"     💡 {issue['recommendation']}")

        # Optimizations
        if analysis_result["optimizations"]:
            print("\n💡 Optimization Suggestions:")
            for opt in analysis_result["optimizations"]:
                print(f"  • {opt['message']}")
                if "recommendation" in opt:
                    print(f"    💡 {opt['recommendation']}")

        # Security concerns
        if analysis_result["security_concerns"]:
            print("\n🔒 Security Concerns:")
            for concern in analysis_result["security_concerns"]:
                icon = "⚠️" if concern["type"] == "warning" else "ℹ️"
                print(f"  {icon} {concern['message']}")
                if "recommendation" in concern:
                    print(f"     💡 {concern['recommendation']}")

        print("\n" + "=" * 60)

    def export_analysis(self, output_path: Union[str, Path]) -> None:
        """Export analysis results to a JSON file."""
        analysis_result = self.analyze_comprehensive()

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)