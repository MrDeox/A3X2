from a3x.planning.mission_planner import MissionPlanner
from a3x.planning.mission_state import (
    Mission,
    MissionMilestone,
    MissionState,
    MetricSnapshot,
)


def build_state(status="active") -> MissionState:
    milestone = MissionMilestone(
        id="m1",
        goal="Alcançar sucesso",
        status="planned",
        capability_tags=["core.diffing"],
        metrics={"core.diffing.success_rate": MetricSnapshot(current=0.0, target=0.9)},
    )
    mission = Mission(
        id="mission-core-diff",
        vision="",
        status=status,
        priority="high",
        milestones=[milestone],
    )
    return MissionState(missions=[mission])


def test_mission_planner_generates_seed() -> None:
    planner = MissionPlanner()
    missions = build_state()
    capability_metrics = {"core.diffing": {"success_rate": 0.5}}

    seeds = planner.propose(
        missions,
        capability_metrics,
        patch_config_path="patch.cfg",
        manual_config_path="manual.cfg",
        tests_config_path="tests.cfg",
        lint_config_path="lint.cfg",
    )

    assert seeds
    seed = seeds[0]
    assert seed.id == "mission.mission-core-diff.m1"
    assert seed.config == "patch.cfg"


def test_mission_planner_skips_completed() -> None:
    planner = MissionPlanner()
    missions = build_state()
    missions.missions[0].milestones[0].status = "completed"
    capability_metrics = {"core.diffing": {"success_rate": 0.95}}

    seeds = planner.propose(
        missions,
        capability_metrics,
        patch_config_path="patch.cfg",
        manual_config_path="manual.cfg",
        tests_config_path="tests.cfg",
        lint_config_path="lint.cfg",
    )

    assert not seeds
