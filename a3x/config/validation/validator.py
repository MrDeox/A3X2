"""Configuration validation engine with strict validation and detailed error reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    jsonschema = None

from a3x.config.validation.schemas import CONFIG_SCHEMA


class ValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(
        self,
        message: str,
        errors: Optional[List[Dict[str, Any]]] = None,
        config_path: Optional[Union[str, Path]] = None
    ):
        super().__init__(message)
        self.errors = errors or []
        self.config_path = Path(config_path) if config_path else None

    def __str__(self) -> str:
        if not self.errors:
            return super().__str__()

        error_lines = [super().__str__()]
        if self.config_path:
            error_lines.append(f"\nConfiguration file: {self.config_path}")

        error_lines.append("\nValidation errors:")
        for i, error in enumerate(self.errors, 1):
            path = " -> ".join(str(p) for p in error.get("absolute_path", []))
            if path:
                error_lines.append(f"  {i}. {path}: {error.get('message', 'Unknown error')}")
            else:
                error_lines.append(f"  {i}. {error.get('message', 'Unknown error')}")

        return "\n".join(error_lines)


def _load_schema(schema_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON schema from file or use embedded schema."""
    if schema_path:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return CONFIG_SCHEMA


def validate_config_structure(
    config_data: Dict[str, Any],
    schema_path: Optional[Path] = None,
    strict: bool = True
) -> None:
    """
    Validate configuration structure against JSON schema.

    Args:
        config_data: Configuration data to validate
        schema_path: Optional path to custom schema file
        strict: If True, raise ValidationError on first failure

    Raises:
        ValidationError: If validation fails
    """
    if not HAS_JSONSCHEMA:
        raise ImportError(
            "jsonschema package is required for configuration validation. "
            "Install it with: pip install jsonschema"
        )

    schema = _load_schema(schema_path)

    # Create validator with detailed error reporting
    validator = jsonschema.Draft7Validator(schema)

    errors = []
    for error in validator.iter_errors(config_data):
        error_info = {
            "message": error.message,
            "absolute_path": list(error.absolute_path),
            "relative_path": list(error.relative_path),
            "schema_path": list(error.schema_path),
        }

        # Add instance value if it's not too large
        try:
            if len(str(error.instance)) < 200:
                error_info["instance"] = error.instance
        except (TypeError, ValueError):
            pass

        errors.append(error_info)

    if errors:
        if strict:
            # For strict validation, raise on first error with full details
            first_error = errors[0]
            message = f"Configuration validation failed: {first_error['message']}"
            raise ValidationError(message, errors)
        else:
            # For non-strict, collect all errors but still raise
            message = f"Configuration validation failed with {len(errors)} error(s)"
            raise ValidationError(message, errors)


def validate_config_value_types(config_data: Dict[str, Any]) -> None:
    """
    Validate individual configuration value types and ranges.

    Args:
        config_data: Configuration data to validate

    Raises:
        ValidationError: If value validation fails
    """
    errors = []

    # Validate LLM configuration
    if "llm" in config_data:
        llm_config = config_data["llm"]

        # Validate LLM type
        if "type" in llm_config:
            valid_types = ["openai", "anthropic", "openrouter", "ollama", "manual"]
            if llm_config["type"] not in valid_types:
                errors.append({
                    "message": f"Invalid LLM type '{llm_config['type']}'. Must be one of: {valid_types}",
                    "absolute_path": ["llm", "type"],
                    "instance": llm_config["type"]
                })

        # Validate model is provided for non-manual types
        if llm_config.get("type") != "manual" and "model" not in llm_config:
            errors.append({
                "message": "LLM model is required for non-manual LLM types",
                "absolute_path": ["llm", "model"]
            })

    # Validate limits configuration
    if "limits" in config_data:
        limits_config = config_data["limits"]

        for limit_field in ["max_iterations", "command_timeout", "max_failures"]:
            if limit_field in limits_config:
                value = limits_config[limit_field]
                if not isinstance(value, int) or value <= 0:
                    errors.append({
                        "message": f"{limit_field} must be a positive integer",
                        "absolute_path": ["limits", limit_field],
                        "instance": value
                    })

    # Validate goals configuration
    if "goals" in config_data:
        goals_config = config_data["goals"]

        for goal_name, goal_value in goals_config.items():
            if isinstance(goal_value, dict) and "min" in goal_value:
                min_value = goal_value["min"]
                if not isinstance(min_value, (int, float)) or not (0.0 <= min_value <= 1.0):
                    errors.append({
                        "message": f"Goal '{goal_name}' min value must be a number between 0.0 and 1.0",
                        "absolute_path": ["goals", goal_name, "min"],
                        "instance": min_value
                    })
            elif isinstance(goal_value, (int, float)):
                if not (0.0 <= goal_value <= 1.0):
                    errors.append({
                        "message": f"Goal '{goal_name}' value must be between 0.0 and 1.0",
                        "absolute_path": ["goals", goal_name],
                        "instance": goal_value
                    })

    if errors:
        message = f"Configuration value validation failed with {len(errors)} error(s)"
        raise ValidationError(message, errors)


def validate_config_paths(config_data: Dict[str, Any], base_dir: Optional[Path] = None) -> None:
    """
    Validate configuration file paths and their existence.

    Args:
        config_data: Configuration data to validate
        base_dir: Base directory for relative path resolution

    Raises:
        ValidationError: If path validation fails
    """
    errors = []

    if base_dir is None:
        base_dir = Path.cwd()

    # Validate workspace root
    if "workspace" in config_data:
        workspace_config = config_data["workspace"]
        if "root" in workspace_config:
            root_path = workspace_config["root"]
            if isinstance(root_path, str):
                full_path = (base_dir / root_path).resolve() if not Path(root_path).is_absolute() else Path(root_path).resolve()

                # Check if path exists (warning only, as it might be created later)
                if not full_path.exists():
                    errors.append({
                        "message": f"Workspace root path does not exist: {full_path}",
                        "absolute_path": ["workspace", "root"],
                        "instance": root_path
                    })

    # Validate loop configuration paths
    if "loop" in config_data:
        loop_config = config_data["loop"]

        for path_field in ["seed_backlog", "seed_config"]:
            if path_field in loop_config:
                path_value = loop_config[path_field]
                if isinstance(path_value, str) and path_value:
                    full_path = (base_dir / path_value).resolve() if not Path(path_value).is_absolute() else Path(path_value).resolve()

                    # Check if path exists for critical files
                    if path_field == "seed_backlog" and not full_path.exists():
                        errors.append({
                            "message": f"Seed backlog file does not exist: {full_path}",
                            "absolute_path": ["loop", path_field],
                            "instance": path_value
                        })

    if errors:
        message = f"Configuration path validation failed with {len(errors)} error(s)"
        raise ValidationError(message, errors)


def validate_config(
    config_data: Dict[str, Any],
    config_path: Optional[Union[str, Path]] = None,
    strict: bool = True,
    validate_paths: bool = True
) -> None:
    """
    Comprehensive configuration validation.

    Args:
        config_data: Configuration data to validate
        config_path: Path to configuration file for path validation
        strict: If True, use strict JSON schema validation
        validate_paths: If True, validate file paths

    Raises:
        ValidationError: If any validation fails
    """
    base_dir = Path(config_path).parent if config_path else None

    # Step 1: JSON Schema validation
    validate_config_structure(config_data, strict=strict)

    # Step 2: Value type validation
    validate_config_value_types(config_data)

    # Step 3: Path validation (if enabled)
    if validate_paths and base_dir:
        validate_config_paths(config_data, base_dir)


def validate_config_file(config_path: Union[str, Path], strict: bool = True) -> None:
    """
    Validate a configuration file from disk.

    Args:
        config_path: Path to configuration file
        strict: If True, use strict validation

    Raises:
        ValidationError: If validation fails
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix.lower() not in [".yaml", ".yml"]:
        raise ValidationError(f"Unsupported configuration file format: {config_path.suffix}")

    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML in configuration file: {e}", config_path=config_path)

    if not isinstance(config_data, dict):
        raise ValidationError("Configuration must be a YAML object/dictionary", config_path=config_path)

    validate_config(config_data, config_path, strict=strict)