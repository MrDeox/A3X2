from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from a3x.actions import ActionType, AgentAction, AgentState, Observation
from a3x.planning.hierarchical_planner import HierarchicalPlanner


def _state(goal: str = "Melhorar docs") -> AgentState:
    return AgentState(goal=goal, history_snapshot="", iteration=1, max_iterations=5)


def test_ensure_plan_persists_structure(tmp_path):
    planner = HierarchicalPlanner(storage_dir=tmp_path)
    plan = planner.ensure_plan(_state(), missions=None, objectives=[], metrics_history={})

    assert plan.current_task == "analisar"
    stored_path = tmp_path / f"{plan.plan_id}.json"
    assert stored_path.exists()
    payload = json.loads(stored_path.read_text(encoding="utf-8"))
    assert payload["missions"][0]["tasks"][0]["status"] == "in_progress"


def test_record_action_and_replan_on_metric_drop(tmp_path):
    planner = HierarchicalPlanner(storage_dir=tmp_path)
    metrics_history = {"actions_success_rate": [1.0]}
    plan = planner.ensure_plan(_state(), missions=None, objectives=[], metrics_history=metrics_history)

    action = AgentAction(type=ActionType.RUN_COMMAND, command=["pytest"])
    observation = Observation(success=True, output="ok")
    planner.record_action_result(action, observation, timestamp=1.0)

    stored_before = json.loads((tmp_path / f"{plan.plan_id}.json").read_text(encoding="utf-8"))
    first_task = stored_before["missions"][0]["tasks"][0]
    assert first_task["status"] == "completed"

    degraded_history = {"actions_success_rate": [1.0, 0.5]}
    replanned = planner.ensure_plan(_state(), missions=None, objectives=[], metrics_history=degraded_history)
    assert replanned.current_task == "analisar"
    stored_after = json.loads((tmp_path / f"{replanned.plan_id}.json").read_text(encoding="utf-8"))
    assert stored_after["missions"][0]["tasks"][0]["status"] == "in_progress"
