"""Persistence helpers for mission state files."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .mission_state import (
    MetricSnapshot,
    Mission,
    MissionMilestone,
    MissionState,
    MissionTelemetry,
)


def _metric_snapshot_from_dict(data: dict[str, object]) -> MetricSnapshot:
    return MetricSnapshot(
        current=float(data.get("current", 0.0)),
        target=float(data["target"]) if data.get("target") is not None else None,
        best=float(data["best"]) if data.get("best") is not None else None,
        samples=int(data.get("samples", 0)),
    )


def _metrics_from_dict(
    data: dict[str, dict[str, object]] | None,
) -> dict[str, MetricSnapshot]:
    metrics: dict[str, MetricSnapshot] = {}
    if not data:
        return metrics
    for name, payload in data.items():
        if isinstance(payload, dict):
            metrics[name] = _metric_snapshot_from_dict(payload)
    return metrics


def _telemetry_from_dict(data: dict[str, object] | None) -> MissionTelemetry:
    if not data:
        return MissionTelemetry()
    telemetry = MissionTelemetry(
        metric_summaries=_metrics_from_dict(data.get("metric_summaries")),
        reflections=list(data.get("reflections", []) or []),
        discovered_gaps=list(data.get("discovered_gaps", []) or []),
        tools_required=list(data.get("tools_required", []) or []),
        last_updated=(
            str(data.get("last_updated"))
            if data.get("last_updated")
            else datetime.now(timezone.utc).isoformat()
        ),
    )
    return telemetry


def _milestone_from_dict(data: dict[str, object]) -> MissionMilestone:
    milestone = MissionMilestone(
        id=str(data["id"]),
        goal=str(data.get("goal", "")),
        status=str(data.get("status", "planned")),
        capability_tags=list(data.get("capability_tags", []) or []),
        dependencies=list(data.get("dependencies", []) or []),
        eta=str(data.get("eta")) if data.get("eta") is not None else None,
        backlog_seed_id=(
            str(data.get("backlog_seed_id"))
            if data.get("backlog_seed_id") is not None
            else None
        ),
        metrics=_metrics_from_dict(data.get("metrics")),
        notes=str(data.get("notes")) if data.get("notes") is not None else None,
    )
    return milestone


def _mission_from_dict(data: dict[str, object]) -> Mission | None:
    try:
        created_at = (
            str(data.get("created_at"))
            if data.get("created_at") is not None
            else datetime.now(timezone.utc).isoformat()
        )
        updated_at = (
            str(data.get("updated_at"))
            if data.get("updated_at") is not None
            else datetime.now(timezone.utc).isoformat()
        )
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    mission = Mission(
        id=str(data["id"]),
        vision=str(data.get("vision", "")),
        status=str(data.get("status", "draft")),
        priority=str(data.get("priority", "medium")),
        success_criteria=list(data.get("success_criteria", []) or []),
        target_metrics=_metrics_from_dict(data.get("target_metrics")),
        capability_tags=list(data.get("capability_tags", []) or []),
        milestones=[],
        telemetry=_telemetry_from_dict(data.get("telemetry")),
        created_at=created_at,
        updated_at=updated_at,
    )

    milestones_raw = data.get("milestones", []) or []
    for milestone_data in milestones_raw:
        if isinstance(milestone_data, dict):
            mission.milestones.append(_milestone_from_dict(milestone_data))
    return mission


def load_mission_state(path: str | Path) -> MissionState:
    path_obj = Path(path)
    if not path_obj.exists():
        return MissionState()
    payload = yaml.safe_load(path_obj.read_text(encoding="utf-8")) or {}
    missions_payload = payload.get("missions", []) if isinstance(payload, dict) else []
    missions = [
        mission
        for item in missions_payload
        if isinstance(item, dict) and (mission := _mission_from_dict(item)) is not None
    ]
    state = MissionState(
        missions=missions,
        generated_at=(
            str(payload.get("generated_at"))
            if isinstance(payload, dict) and payload.get("generated_at")
            else datetime.now(timezone.utc).isoformat()
        ),
        version=(
            str(payload.get("version", "0.1")) if isinstance(payload, dict) else "0.1"
        ),
    )
    return state


def save_mission_state(state: MissionState, path: str | Path) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_dict()
    path_obj.write_text(
        yaml.safe_dump(payload, allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )


def sync_mission_state(
    state: MissionState, missions: Iterable[Mission]
) -> MissionState:
    idx = {mission.id: mission for mission in state.missions}
    for mission in missions:
        idx[mission.id] = mission
    state.missions = list(idx.values())
    state.generated_at = datetime.now(timezone.utc).isoformat()
    return state


__all__ = ["load_mission_state", "save_mission_state", "sync_mission_state"]
