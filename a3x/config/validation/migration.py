"""Configuration migration framework for backward compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Version

from a3x.config.validation.schemas import CONFIG_SCHEMA


class MigrationStep:
    """Represents a single configuration migration step."""

    def __init__(
        self,
        version_from: str,
        version_to: str,
        description: str,
        migration_func: callable
    ):
        self.version_from = version_from
        self.version_to = version_to
        self.description = description
        self.migration_func = migration_func

    def can_apply(self, current_version: str) -> bool:
        """Check if this migration can be applied to the current version."""
        return current_version == self.version_from

    def apply(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this migration step to the configuration."""
        return self.migration_func(config)


class ConfigurationMigrator:
    """Handles configuration migrations across versions."""

    def __init__(self):
        self.migration_steps: List[MigrationStep] = []
        self._register_default_migrations()

    def _register_default_migrations(self):
        """Register default migration steps."""

        # Migration from v1.0 to v1.1: Rename fields and add defaults
        def migrate_v1_to_v1_1(config: Dict[str, Any]) -> Dict[str, Any]:
            """Migrate from v1.0 to v1.1."""
            migrated = config.copy()

            # Rename 'max_iterations' to 'max_iterations' (no change needed)
            # Add new 'scaling' section with defaults
            if 'scaling' not in migrated:
                migrated['scaling'] = {
                    'cpu_threshold': 0.8,
                    'memory_threshold': 0.8,
                    'max_recursion_adjust': 3
                }

            # Add 'audit' section if missing
            if 'audit' not in migrated:
                migrated['audit'] = {
                    'enable_file_log': True,
                    'file_dir': 'seed/changes',
                    'enable_git_commit': False,
                    'commit_prefix': 'A3X'
                }

            return migrated

        # Migration from v1.1 to v1.2: Restructure goals format
        def migrate_v1_1_to_v1_2(config: Dict[str, Any]) -> Dict[str, Any]:
            """Migrate from v1.1 to v1.2."""
            migrated = config.copy()

            # Restructure goals to support both formats
            if 'goals' in migrated:
                goals = migrated['goals']
                for goal_name, goal_value in goals.items():
                    if isinstance(goal_value, (int, float)):
                        # Convert simple number to object format for clarity
                        goals[goal_name] = {'min': float(goal_value)}

            return migrated

        # Migration from v1.2 to v1.3: Add new policy options
        def migrate_v1_2_to_v1_3(config: Dict[str, Any]) -> Dict[str, Any]:
            """Migrate from v1.2 to v1.3."""
            migrated = config.copy()

            # Add new policy fields with secure defaults
            if 'policies' in migrated:
                policies = migrated['policies']
                if 'allow_shell_write' not in policies:
                    policies['allow_shell_write'] = True

            return migrated

        # Register migration steps
        self.add_migration("1.0.0", "1.1.0", "Add scaling and audit sections", migrate_v1_to_v1_1)
        self.add_migration("1.1.0", "1.2.0", "Restructure goals format", migrate_v1_1_to_v1_2)
        self.add_migration("1.2.0", "1.3.0", "Add shell write policy", migrate_v1_2_to_v1_3)

    def add_migration(
        self,
        version_from: str,
        version_to: str,
        description: str,
        migration_func: callable
    ):
        """Add a new migration step."""
        step = MigrationStep(version_from, version_to, description, migration_func)
        self.migration_steps.append(step)

    def get_migration_path(self, current_version: str, target_version: str) -> List[MigrationStep]:
        """
        Get the migration path from current version to target version.

        Args:
            current_version: Current configuration version
            target_version: Target configuration version

        Returns:
            List of migration steps to apply
        """
        applicable_steps = []

        # Simple version comparison (can be enhanced for semantic versioning)
        def version_key(version: str) -> Tuple[int, ...]:
            return tuple(int(x) for x in version.split('.'))

        current_key = version_key(current_version)
        target_key = version_key(target_version)

        if current_key >= target_key:
            return []  # Already at or beyond target version

        # Find applicable migration steps
        for step in self.migration_steps:
            step_key = version_key(step.version_from)
            if current_key <= step_key < target_key:
                applicable_steps.append(step)

        # Sort by version order
        applicable_steps.sort(key=lambda s: version_key(s.version_from))

        return applicable_steps

    def migrate_config(
        self,
        config: Dict[str, Any],
        current_version: str = "1.0.0",
        target_version: str = "1.3.0"
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Migrate configuration from current version to target version.

        Args:
            config: Configuration to migrate
            current_version: Current configuration version
            target_version: Target configuration version

        Returns:
            Tuple of (migrated_config, migration_messages)
        """
        migration_steps = self.get_migration_path(current_version, target_version)
        migrated_config = config.copy()
        messages = []

        for step in migration_steps:
            try:
                messages.append(f"Applying migration: {step.description}")
                migrated_config = step.apply(migrated_config)
            except Exception as e:
                messages.append(f"Failed to apply migration {step.description}: {e}")
                raise

        return migrated_config, messages

    def detect_version(self, config: Dict[str, Any]) -> str:
        """
        Detect the version of a configuration based on its structure.

        Args:
            config: Configuration to analyze

        Returns:
            Detected version string
        """
        # Simple version detection based on presence of certain fields
        has_scaling = 'scaling' in config
        has_audit = 'audit' in config

        if has_scaling and has_audit:
            # Check if goals are in object format (v1.2+)
            if 'goals' in config:
                goals = config['goals']
                sample_goal = next(iter(goals.values()), None) if goals else None
                if isinstance(sample_goal, dict) and 'min' in sample_goal:
                    return "1.3.0"
                else:
                    return "1.2.0"
            return "1.2.0"
        elif has_scaling or has_audit:
            return "1.1.0"
        else:
            return "1.0.0"

    def create_backup(self, config: Dict[str, Any], config_path: Path) -> Path:
        """Create a backup of the original configuration."""
        timestamp = Path(config_path).stem + "_backup_" + str(int(Path(".").stat().st_mtime))
        backup_path = config_path.with_suffix(f".{timestamp}.yaml")

        import yaml
        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)

        return backup_path

    def migrate_file(
        self,
        config_path: Path,
        current_version: Optional[str] = None,
        target_version: str = "1.3.0",
        create_backup: bool = True
    ) -> Tuple[Dict[str, Any], List[str], Optional[Path]]:
        """
        Migrate a configuration file.

        Args:
            config_path: Path to configuration file
            current_version: Current version (auto-detected if None)
            target_version: Target version
            create_backup: Whether to create backup before migration

        Returns:
            Tuple of (migrated_config, messages, backup_path)
        """
        import yaml

        # Read original config
        with open(config_path, 'r', encoding='utf-8') as f:
            original_config = yaml.safe_load(f) or {}

        # Detect version if not provided
        if current_version is None:
            current_version = self.detect_version(original_config)

        # Create backup if requested
        backup_path = None
        if create_backup:
            backup_path = self.create_backup(original_config, config_path)

        # Migrate configuration
        migrated_config, messages = self.migrate_config(
            original_config,
            current_version,
            target_version
        )

        # Write migrated config back
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(migrated_config, f, default_flow_style=False, indent=2)

        return migrated_config, messages, backup_path


# Global migrator instance
_config_migrator = ConfigurationMigrator()


def migrate_config(
    config: Dict[str, Any],
    current_version: str = "1.0.0",
    target_version: str = "1.3.0"
) -> Tuple[Dict[str, Any], List[str]]:
    """Migrate configuration to target version."""
    return _config_migrator.migrate_config(config, current_version, target_version)


def migrate_config_file(
    config_path: Path,
    current_version: Optional[str] = None,
    target_version: str = "1.3.0",
    create_backup: bool = True
) -> Tuple[Dict[str, Any], List[str], Optional[Path]]:
    """Migrate a configuration file."""
    return _config_migrator.migrate_file(config_path, current_version, target_version, create_backup)