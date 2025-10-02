"""Configuration manager for autonomous goal generation system.

This module provides configuration management for triggers, thresholds,
motivation profiles, and integration settings for the autonomous goal
generation system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AutonomousGoalTriggers:
    """Configuration for triggers that activate autonomous goal generation."""

    # Thresholds for triggering different types of goals
    curiosity_threshold: float = 0.6
    performance_degradation_threshold: float = 0.8
    coverage_threshold: float = 0.7
    novelty_threshold: float = 0.5

    # Minimum intervals between goal generation (in hours)
    min_interval_between_runs: int = 2
    min_interval_between_similar_goals: int = 24

    # Performance thresholds for triggering optimization goals
    min_performance_drop_for_optimization: float = 0.1  # 10% drop
    min_success_rate_for_gap_goals: float = 0.8

    def should_trigger_curiosity_goal(self, curiosity_score: float) -> bool:
        """Check if curiosity score meets threshold for triggering goal generation."""
        return curiosity_score >= self.curiosity_threshold

    def should_trigger_performance_goal(self, current_performance: float, baseline: float) -> bool:
        """Check if performance degradation should trigger optimization goal."""
        if baseline == 0:
            return False
        degradation = (baseline - current_performance) / baseline
        return degradation >= self.min_performance_drop_for_optimization

    def should_trigger_coverage_goal(self, coverage_score: float) -> bool:
        """Check if coverage score is low enough to trigger improvement goal."""
        return coverage_score <= self.coverage_threshold

    def should_trigger_novelty_goal(self, novelty_score: float) -> bool:
        """Check if novelty score meets threshold for triggering exploration goal."""
        return novelty_score >= self.novelty_threshold


@dataclass
class MotivationProfile:
    """Profile for intrinsic motivation factors in goal generation."""

    curiosity_weight: float = 0.3
    competence_weight: float = 0.25
    autonomy_weight: float = 0.2
    relatedness_weight: float = 0.15
    exploration_bias: float = 0.1

    # Specific thresholds for curiosity-driven exploration
    novelty_threshold: float = 0.6
    uncertainty_tolerance: float = 0.4
    domain_diversity_factor: float = 0.3

    def calculate_total_motivation(self, factors: dict[str, float]) -> float:
        """Calculate total motivation score from individual factors."""
        total = 0.0
        for factor_name, factor_value in factors.items():
            if hasattr(self, f"{factor_name}_weight"):
                weight = getattr(self, f"{factor_name}_weight")
                total += factor_value * weight

        # Add exploration bias if applicable
        if "exploration_value" in factors:
            total += factors["exploration_value"] * self.exploration_bias

        return min(total, 1.0)  # Cap at 1.0

    def should_explore_domain(self, novelty_score: float, uncertainty_score: float) -> bool:
        """Determine if a domain should be explored based on curiosity factors."""
        return (novelty_score >= self.novelty_threshold and
                uncertainty_score >= self.uncertainty_tolerance)


@dataclass
class IntegrationSettings:
    """Settings for integrating autonomous goals with existing systems."""

    # Maximum number of autonomous goals to generate per cycle
    max_goals_per_cycle: int = 10

    # Prioritization settings
    prioritize_recent_gaps: bool = True
    prioritize_high_impact: bool = True
    prioritize_curiosity_goals: bool = False

    # Seed backlog settings
    autonomous_goals_backlog: str = "seed/autonomous_goals_backlog.yaml"
    enable_autonomous_goal_backlog: bool = True
    main_backlog: str = "seed/backlog.yaml"

    # Generation control
    generation_interval_seconds: int = 300  # 5 minutes
    max_consecutive_failures: int = 3

    def get_prioritization_weights(self) -> dict[str, float]:
        """Get prioritization weights for different goal types."""
        weights = {
            "recency": 0.2 if self.prioritize_recent_gaps else 0.1,
            "impact": 0.4 if self.prioritize_high_impact else 0.2,
            "curiosity": 0.3 if self.prioritize_curiosity_goals else 0.2,
            "type_diversity": 0.1
        }

        # Normalize weights
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


@dataclass
class MonitoringSettings:
    """Settings for monitoring and evaluating autonomous goal generation."""

    enable_goal_generation_metrics: bool = True
    metrics_file: str = "seed/metrics/autonomous_goals_history.json"

    track_goal_success_rate: bool = True
    track_motivation_effectiveness: bool = True
    track_exploration_diversity: bool = True

    # Evaluation intervals (in cycles)
    evaluation_interval: int = 5
    reflection_interval: int = 10

    # Alerting thresholds
    low_success_rate_threshold: float = 0.5
    high_failure_rate_threshold: float = 0.3

    def should_run_evaluation(self, cycle_count: int) -> bool:
        """Check if evaluation should run based on cycle count."""
        return cycle_count % self.evaluation_interval == 0

    def should_run_reflection(self, cycle_count: int) -> bool:
        """Check if meta-reflection should run based on cycle count."""
        return cycle_count % self.reflection_interval == 0


@dataclass
class AutonomousGoalConfig:
    """Main configuration class for autonomous goal generation system."""

    # Core settings
    enable: bool = True
    config_file: str = "configs/seed_autonomous_goals.yaml"

    # Sub-configurations
    triggers: AutonomousGoalTriggers = field(default_factory=AutonomousGoalTriggers)
    motivation_profile: MotivationProfile = field(default_factory=MotivationProfile)
    integration: IntegrationSettings = field(default_factory=IntegrationSettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)

    # Goal type specific settings
    goal_type_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> AutonomousGoalConfig:
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            # Return default configuration
            return cls()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
        except Exception:
            return cls()

        # Extract main settings
        autonomous_goals_config = config_data.get("autonomous_goals", {})

        # Build configuration object
        config = cls(
            enable=autonomous_goals_config.get("enable", True),
            config_file=autonomous_goals_config.get("config_file", "configs/seed_autonomous_goals.yaml")
        )

        # Load triggers configuration
        if "triggers" in autonomous_goals_config:
            triggers_data = autonomous_goals_config["triggers"]
            config.triggers = AutonomousGoalTriggers(
                curiosity_threshold=triggers_data.get("curiosity_threshold", 0.6),
                performance_degradation_threshold=triggers_data.get("performance_degradation_threshold", 0.8),
                coverage_threshold=triggers_data.get("coverage_threshold", 0.7),
                novelty_threshold=triggers_data.get("novelty_threshold", 0.5),
                min_interval_between_runs=triggers_data.get("min_interval_between_runs", 2),
                min_interval_between_similar_goals=triggers_data.get("min_interval_between_similar_goals", 24)
            )

        # Load motivation profile
        if "motivation_profile" in autonomous_goals_config:
            profile_data = autonomous_goals_config["motivation_profile"]
            config.motivation_profile = MotivationProfile(
                curiosity_weight=profile_data.get("curiosity_weight", 0.3),
                competence_weight=profile_data.get("competence_weight", 0.25),
                autonomy_weight=profile_data.get("autonomy_weight", 0.2),
                relatedness_weight=profile_data.get("relatedness_weight", 0.15),
                exploration_bias=profile_data.get("exploration_bias", 0.1),
                novelty_threshold=profile_data.get("novelty_threshold", 0.6),
                uncertainty_tolerance=profile_data.get("uncertainty_tolerance", 0.4),
                domain_diversity_factor=profile_data.get("domain_diversity_factor", 0.3)
            )

        # Load integration settings
        if "integration" in autonomous_goals_config:
            integration_data = autonomous_goals_config["integration"]
            config.integration = IntegrationSettings(
                max_goals_per_cycle=integration_data.get("max_goals_per_cycle", 10),
                prioritize_recent_gaps=integration_data.get("prioritize_recent_gaps", True),
                prioritize_high_impact=integration_data.get("prioritize_high_impact", True),
                prioritize_curiosity_goals=integration_data.get("prioritize_curiosity_goals", False),
                autonomous_goals_backlog=integration_data.get("autonomous_goals_backlog", "seed/autonomous_goals_backlog.yaml"),
                enable_autonomous_goal_backlog=integration_data.get("enable_autonomous_goal_backlog", True),
                generation_interval_seconds=integration_data.get("generation_interval_seconds", 300)
            )

        # Load monitoring settings
        if "monitoring" in autonomous_goals_config:
            monitoring_data = autonomous_goals_config["monitoring"]
            config.monitoring = MonitoringSettings(
                enable_goal_generation_metrics=monitoring_data.get("enable_goal_generation_metrics", True),
                metrics_file=monitoring_data.get("metrics_file", "seed/metrics/autonomous_goals_history.json"),
                track_goal_success_rate=monitoring_data.get("track_goal_success_rate", True),
                track_motivation_effectiveness=monitoring_data.get("track_motivation_effectiveness", True),
                track_exploration_diversity=monitoring_data.get("track_exploration_diversity", True),
                evaluation_interval=monitoring_data.get("evaluation_interval", 5),
                reflection_interval=monitoring_data.get("reflection_interval", 10)
            )

        return config

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """Save generation metrics if monitoring is enabled."""
        if not self.monitoring.enable_goal_generation_metrics:
            return

        try:
            metrics_file = Path(self.monitoring.metrics_file)
            metrics_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing metrics
            history = []
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                except Exception:
                    history = []

            # Add new metrics with timestamp
            metrics_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "triggers": {
                        "curiosity_threshold": self.triggers.curiosity_threshold,
                        "performance_degradation_threshold": self.triggers.performance_degradation_threshold,
                        "coverage_threshold": self.triggers.coverage_threshold,
                        "novelty_threshold": self.triggers.novelty_threshold
                    },
                    "motivation_profile": {
                        "curiosity_weight": self.motivation_profile.curiosity_weight,
                        "competence_weight": self.motivation_profile.competence_weight,
                        "autonomy_weight": self.motivation_profile.autonomy_weight
                    },
                    "integration": {
                        "max_goals_per_cycle": self.integration.max_goals_per_cycle,
                        "prioritize_high_impact": self.integration.prioritize_high_impact,
                        "prioritize_curiosity_goals": self.integration.prioritize_curiosity_goals
                    }
                },
                "metrics": metrics
            }

            history.append(metrics_entry)

            # Keep only last 1000 entries
            if len(history) > 1000:
                history = history[-1000:]

            # Save updated history
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        except Exception:
            # Silently fail if metrics can't be saved
            pass

    def should_generate_goals(self, last_generation_time: datetime | None = None) -> bool:
        """Check if goals should be generated based on timing and triggers."""
        if not self.enable:
            return False

        # Check timing constraint
        if last_generation_time:
            elapsed_seconds = (datetime.now(timezone.utc) - last_generation_time).total_seconds()
            if elapsed_seconds < self.integration.generation_interval_seconds:
                return False

        return True


def get_autonomous_goal_config(workspace_root: Path | None = None) -> AutonomousGoalConfig:
    """Get autonomous goal configuration from standard locations."""
    if workspace_root is None:
        workspace_root = Path.cwd()

    # Try to load from main config file first
    main_config = workspace_root / "configs" / "sample.yaml"
    if main_config.exists():
        try:
            # Load main config to check for autonomous_goals settings
            with open(main_config, 'r', encoding='utf-8') as f:
                main_data = yaml.safe_load(f) or {}

            autonomous_config_file = main_data.get("autonomous_goals", {}).get("config_file")
            if autonomous_config_file:
                config_path = workspace_root / autonomous_config_file
                if config_path.exists():
                    return AutonomousGoalConfig.from_yaml(config_path)
        except Exception:
            pass

    # Fall back to default config file
    default_config = workspace_root / "configs" / "seed_autonomous_goals.yaml"
    return AutonomousGoalConfig.from_yaml(default_config)


__all__ = [
    "AutonomousGoalConfig",
    "AutonomousGoalTriggers",
    "MotivationProfile",
    "IntegrationSettings",
    "MonitoringSettings",
    "get_autonomous_goal_config"
]