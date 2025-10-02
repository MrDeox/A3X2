"""Configuration schemas for A3X."""

import json
from pathlib import Path
from typing import Dict, Any


# Embedded JSON schema as fallback
CONFIG_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://a3x.dev/config-schema.json",
    "title": "A3X Agent Configuration",
    "description": "Complete configuration schema for A3X autonomous agent",
    "type": "object",
    "properties": {
        "llm": {
            "$ref": "#/$defs/llm_config"
        },
        "workspace": {
            "$ref": "#/$defs/workspace_config"
        },
        "limits": {
            "$ref": "#/$defs/limits_config"
        },
        "tests": {
            "$ref": "#/$defs/tests_config"
        },
        "policies": {
            "$ref": "#/$defs/policies_config"
        },
        "goals": {
            "$ref": "#/$defs/goals_config"
        },
        "loop": {
            "$ref": "#/$defs/loop_config"
        },
        "audit": {
            "$ref": "#/$defs/audit_config"
        },
        "scaling": {
            "$ref": "#/$defs/scaling_config"
        }
    },
    "required": ["llm", "workspace"],
    "additionalProperties": False,
    "$defs": {
        "llm_config": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["openai", "anthropic", "openrouter", "ollama", "manual"],
                    "description": "LLM provider type"
                },
                "model": {
                    "type": "string",
                    "description": "Model name or identifier"
                },
                "script": {
                    "type": "string",
                    "description": "Path to custom script for manual LLM"
                },
                "endpoint": {
                    "type": "string",
                    "format": "uri",
                    "description": "API endpoint URL"
                },
                "api_key_env": {
                    "type": "string",
                    "description": "Environment variable name for API key"
                },
                "base_url": {
                    "type": "string",
                    "format": "uri",
                    "description": "Base URL for API requests"
                }
            },
            "required": ["type"],
            "additionalProperties": False
        },
        "workspace_config": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Workspace root directory path"
                },
                "allow_outside_root": {
                    "type": "boolean",
                    "description": "Allow operations outside workspace root"
                }
            },
            "additionalProperties": False
        },
        "limits_config": {
            "type": "object",
            "properties": {
                "max_iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "description": "Maximum number of iterations"
                },
                "command_timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3600,
                    "description": "Command execution timeout in seconds"
                },
                "max_failures": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of allowed failures"
                },
                "total_timeout": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 86400,
                    "description": "Total execution timeout in seconds"
                }
            },
            "additionalProperties": False
        },
        "tests_config": {
            "type": "object",
            "properties": {
                "auto": {
                    "type": "boolean",
                    "description": "Enable automatic test execution"
                },
                "commands": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {
                                "type": "string"
                            },
                            {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        ]
                    },
                    "description": "Test commands to execute"
                }
            },
            "additionalProperties": False
        },
        "policies_config": {
            "type": "object",
            "properties": {
                "allow_network": {
                    "type": "boolean",
                    "description": "Allow network access"
                },
                "allow_shell_write": {
                    "type": "boolean",
                    "description": "Allow shell write operations"
                },
                "deny_commands": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Commands to deny execution"
                }
            },
            "additionalProperties": False
        },
        "goals_config": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z_][a-zA-Z0-9_]*$": {
                    "oneOf": [
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        {
                            "type": "object",
                            "properties": {
                                "min": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                }
                            },
                            "required": ["min"],
                            "additionalProperties": False
                        }
                    ]
                }
            },
            "additionalProperties": False
        },
        "loop_config": {
            "type": "object",
            "properties": {
                "auto_seed": {
                    "type": "boolean",
                    "description": "Enable automatic seed generation"
                },
                "seed_backlog": {
                    "type": "string",
                    "description": "Path to seed backlog file"
                },
                "seed_config": {
                    "type": "string",
                    "description": "Path to seed configuration file"
                },
                "seed_interval": {
                    "type": "number",
                    "minimum": 0.0,
                    "description": "Interval between seed generation in seconds"
                },
                "seed_max_runs": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": "Maximum number of seed runs"
                },
                "stop_when_idle": {
                    "type": "boolean",
                    "description": "Stop execution when idle"
                },
                "use_memory": {
                    "type": "boolean",
                    "description": "Enable memory usage"
                },
                "memory_top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Number of top memories to retrieve"
                },
                "interactive": {
                    "type": "boolean",
                    "description": "Enable interactive mode"
                }
            },
            "additionalProperties": False
        },
        "audit_config": {
            "type": "object",
            "properties": {
                "enable_file_log": {
                    "type": "boolean",
                    "description": "Enable file-based logging"
                },
                "file_dir": {
                    "type": "string",
                    "description": "Directory for audit files"
                },
                "enable_git_commit": {
                    "type": "boolean",
                    "description": "Enable git commits for changes"
                },
                "commit_prefix": {
                    "type": "string",
                    "description": "Prefix for git commit messages"
                }
            },
            "additionalProperties": False
        },
        "scaling_config": {
            "type": "object",
            "properties": {
                "cpu_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "CPU usage threshold for scaling"
                },
                "memory_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Memory usage threshold for scaling"
                },
                "max_recursion_adjust": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum recursion adjustments"
                }
            },
            "additionalProperties": false
        }
    }
}


def load_schema(schema_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load schema from file or return embedded schema."""
    if schema_path and schema_path.exists():
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return CONFIG_SCHEMA