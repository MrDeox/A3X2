"""Configuration generation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from a3x.config.validation.schemas import CONFIG_SCHEMA


class ConfigGenerator:
    """Generates configuration files and templates."""

    def __init__(self):
        self.schema = CONFIG_SCHEMA

    def generate_minimal_config(self, output_path: Union[str, Path]) -> None:
        """Generate a minimal configuration file."""
        config = {
            "llm": {
                "type": "manual",
                "script": "echo 'Hello from A3X'"
            },
            "workspace": {
                "root": ".",
                "allow_outside_root": False
            }
        }

        self._write_config(config, output_path)

    def generate_development_config(self, output_path: Union[str, Path]) -> None:
        """Generate a development-ready configuration file."""
        config = {
            "llm": {
                "type": "openrouter",
                "model": "x-ai/grok-4-fast:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY"
            },
            "workspace": {
                "root": ".",
                "allow_outside_root": True
            },
            "limits": {
                "max_iterations": 25,
                "command_timeout": 120,
                "max_failures": 5
            },
            "tests": {
                "auto": False,
                "commands": []
            },
            "policies": {
                "allow_network": True,
                "allow_shell_write": True,
                "deny_commands": ["rm -rf", "sudo", "su"]
            },
            "goals": {
                "apply_patch_success_rate": {"min": 0.9},
                "actions_success_rate": {"min": 0.85},
                "tests_success_rate": {"min": 0.95}
            },
            "loop": {
                "auto_seed": True,
                "seed_backlog": "seed/backlog.yaml",
                "seed_config": "configs/seed_manual.yaml",
                "seed_interval": 0,
                "stop_when_idle": True,
                "use_memory": False,
                "memory_top_k": 3
            },
            "audit": {
                "enable_file_log": True,
                "file_dir": "seed/changes",
                "enable_git_commit": True,
                "commit_prefix": "A3X"
            },
            "scaling": {
                "cpu_threshold": 0.8,
                "memory_threshold": 0.8,
                "max_recursion_adjust": 3
            }
        }

        self._write_config(config, output_path)

    def generate_secure_config(self, output_path: Union[str, Path]) -> None:
        """Generate a security-hardened configuration file."""
        config = {
            "llm": {
                "type": "manual",
                "script": "echo 'Secure mode: manual only'"
            },
            "workspace": {
                "root": ".",
                "allow_outside_root": False
            },
            "limits": {
                "max_iterations": 10,
                "command_timeout": 60,
                "max_failures": 3,
                "total_timeout": 1800
            },
            "tests": {
                "auto": False,
                "commands": []
            },
            "policies": {
                "allow_network": False,
                "allow_shell_write": False,
                "deny_commands": [
                    "rm", "rmdir", "del", "format", "fdisk",
                    "sudo", "su", "chmod", "chown", "passwd",
                    "usermod", "userdel", "groupmod", "mount",
                    "umount", "systemctl", "service", "kill",
                    "killall", "pkill", "wget", "curl", "nc"
                ]
            },
            "goals": {
                "apply_patch_success_rate": {"min": 0.95},
                "actions_success_rate": {"min": 0.9},
                "tests_success_rate": {"min": 0.98}
            },
            "loop": {
                "auto_seed": False,
                "seed_backlog": "seed/backlog.yaml",
                "seed_config": None,
                "seed_interval": 0,
                "stop_when_idle": True,
                "use_memory": False,
                "memory_top_k": 1
            },
            "audit": {
                "enable_file_log": True,
                "file_dir": "seed/changes",
                "enable_git_commit": False,
                "commit_prefix": "A3X"
            },
            "scaling": {
                "cpu_threshold": 0.5,
                "memory_threshold": 0.5,
                "max_recursion_adjust": 1
            }
        }

        self._write_config(config, output_path)

    def generate_llm_configs(self, output_dir: Union[str, Path]) -> None:
        """Generate example configurations for different LLM providers."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # OpenAI configuration
        openai_config = {
            "llm": {
                "type": "openai",
                "model": "gpt-4",
                "api_key_env": "OPENAI_API_KEY"
            },
            "workspace": {"root": ".", "allow_outside_root": False},
            "limits": {"max_iterations": 50, "command_timeout": 120, "max_failures": 10},
            "policies": {"allow_network": False, "allow_shell_write": True},
            "goals": {"apply_patch_success_rate": {"min": 0.9}}
        }
        self._write_config(openai_config, output_dir / "openai_config.yaml")

        # Anthropic configuration
        anthropic_config = {
            "llm": {
                "type": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "api_key_env": "ANTHROPIC_API_KEY"
            },
            "workspace": {"root": ".", "allow_outside_root": False},
            "limits": {"max_iterations": 50, "command_timeout": 120, "max_failures": 10},
            "policies": {"allow_network": False, "allow_shell_write": True},
            "goals": {"apply_patch_success_rate": {"min": 0.9}}
        }
        self._write_config(anthropic_config, output_dir / "anthropic_config.yaml")

        # OpenRouter configuration
        openrouter_config = {
            "llm": {
                "type": "openrouter",
                "model": "x-ai/grok-4-fast:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY"
            },
            "workspace": {"root": ".", "allow_outside_root": False},
            "limits": {"max_iterations": 50, "command_timeout": 120, "max_failures": 10},
            "policies": {"allow_network": True, "allow_shell_write": True},
            "goals": {"apply_patch_success_rate": {"min": 0.9}}
        }
        self._write_config(openrouter_config, output_dir / "openrouter_config.yaml")

        # Ollama configuration
        ollama_config = {
            "llm": {
                "type": "ollama",
                "model": "llama2",
                "base_url": "http://localhost:11434",
                "endpoint": "http://localhost:11434/api/generate"
            },
            "workspace": {"root": ".", "allow_outside_root": False},
            "limits": {"max_iterations": 50, "command_timeout": 120, "max_failures": 10},
            "policies": {"allow_network": True, "allow_shell_write": True},
            "goals": {"apply_patch_success_rate": {"min": 0.9}}
        }
        self._write_config(ollama_config, output_dir / "ollama_config.yaml")

    def _write_config(self, config: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Write configuration to file."""
        import yaml

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# A3X Configuration File\n")
            f.write(f"# Generated: {output_path}\n")
            f.write("# Description: Auto-generated configuration template\n\n")
            yaml.safe_dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)

    def generate_schema_docs(self, output_path: Union[str, Path]) -> None:
        """Generate documentation from JSON schema."""
        output_path = Path(output_path)

        docs = self._generate_schema_documentation()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(docs)

    def _generate_schema_documentation(self) -> str:
        """Generate markdown documentation from schema."""
        lines = [
            "# A3X Configuration Schema Documentation",
            "",
            "This document describes the complete configuration schema for A3X autonomous agent.",
            "",
            "## Configuration Sections",
            ""
        ]

        # Document each section
        for section_name, section_schema in self.schema.get("properties", {}).items():
            lines.extend(self._document_section(section_name, section_schema))

        return "\n".join(lines)

    def _document_section(self, section_name: str, section_schema: Dict[str, Any]) -> List[str]:
        """Document a single configuration section."""
        lines = [
            f"### {section_name.title()} Configuration",
            "",
            f"**Description**: {section_schema.get('description', 'No description available')}",
            ""
        ]

        # Document properties
        if "properties" in section_schema:
            lines.append("**Properties**:")
            lines.append("")

            for prop_name, prop_schema in section_schema["properties"].items():
                lines.extend(self._document_property(prop_name, prop_schema))

        lines.append("")
        return lines

    def _document_property(self, prop_name: str, prop_schema: Dict[str, Any]) -> List[str]:
        """Document a single property."""
        lines = [
            f"- **{prop_name}**",
            f"  - Type: `{prop_schema.get('type', 'unknown')}`",
        ]

        if "description" in prop_schema:
            lines.append(f"  - Description: {prop_schema['description']}")

        if "enum" in prop_schema:
            lines.append(f"  - Allowed values: {', '.join(f'`{v}`' for v in prop_schema['enum'])}")

        if "minimum" in prop_schema:
            lines.append(f"  - Minimum: {prop_schema['minimum']}")

        if "maximum" in prop_schema:
            lines.append(f"  - Maximum: {prop_schema['maximum']}")

        if "default" in prop_schema:
            lines.append(f"  - Default: `{prop_schema['default']}`")

        if prop_schema.get("required", False):
            lines.append("  - **Required**: Yes")

        lines.append("")
        return lines