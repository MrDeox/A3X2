"""Planner that transforms mission milestones into concrete seeds."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from ..seeds import Seed
from .mission_state import MissionState, MissionMilestone


_CAPABILITY_DEFAULT_CONFIG = {
    "core.diffing": "patch",
    "core.testing": "tests",
    "horiz.python": "manual",
    "horiz.docs": "manual",
}


class MissionPlanner:
    """Translate mission milestones into actionable seeds."""

    def propose(
        self,
        missions: MissionState,
        capability_metrics: Dict[str, Dict[str, float | int | None]],
        *,
        patch_config_path: str,
        manual_config_path: str,
        tests_config_path: str,
        lint_config_path: str,
    ) -> List[Seed]:
        seeds: List[Seed] = []

        config_map = {
            "patch": patch_config_path,
            "tests": tests_config_path,
            "lint": lint_config_path,
            "manual": manual_config_path,
        }

        for mission in missions.missions:
            if mission.status == "completed":
                continue

            for milestone in mission.milestones:
                if milestone.status == "completed":
                    continue

                if milestone.dependencies and not all(
                    _milestone_completed(mission.milestones, dep)
                    for dep in milestone.dependencies
                ):
                    continue

                if milestone.metrics:
                    all_met = True
                    for metric_name, snapshot in milestone.metrics.items():
                        value = _resolve_metric(metric_name, capability_metrics)
                        if value is None:
                            all_met = False
                            break
                        target = snapshot.target
                        if target is not None and value < target:
                            all_met = False
                            break
                    if all_met:
                        continue

                seed_id = f"mission.{mission.id}.{milestone.id}"
                config = self._resolve_config(milestone, config_map)
                seed = Seed(
                    id=seed_id,
                    goal=milestone.goal or f"Avançar missão {mission.id}",
                    priority=(
                        "high" if mission.priority in {"high", "moonshot"} else "medium"
                    ),
                    status="pending",
                    type="mission",
                    config=config,
                    max_steps=6,
                    metadata={
                        "mission": mission.id,
                        "milestone": milestone.id,
                        "description": milestone.goal,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                seeds.append(seed)
        return seeds

    def _resolve_config(
        self, milestone: MissionMilestone, config_map: Dict[str, str]
    ) -> str:
        for capability in milestone.capability_tags:
            key = _CAPABILITY_DEFAULT_CONFIG.get(capability, "manual")
            if key in config_map:
                return config_map[key]
        return config_map["manual"]


def _milestone_completed(milestones: List[MissionMilestone], milestone_id: str) -> bool:
    for milestone in milestones:
        if milestone.id == milestone_id:
            return milestone.status == "completed"
    return False


def _resolve_metric(
    metric_ref: str, capability_metrics: Dict[str, Dict[str, float | int | None]]
) -> float | None:
    if not metric_ref:
        return None
    if "." in metric_ref:
        capability, metric = metric_ref.rsplit(".", 1)
    else:
        capability, metric = metric_ref, ""
    data = capability_metrics.get(capability)
    if not data:
        return None
    if metric:
        value = data.get(metric)
    else:
        value = None
    if isinstance(value, (int, float)):
        return float(value)
    return None
