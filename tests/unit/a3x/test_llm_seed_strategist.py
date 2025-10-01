from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from a3x.actions import ActionType, AgentAction, Observation
from a3x.llm_seed_strategist import LLMSeedStrategist


def test_capture_failure_generates_seed(tmp_path):
    strategist = LLMSeedStrategist(backlog_path=tmp_path / "backlog.yaml")
    action = AgentAction(type=ActionType.RUN_COMMAND, command=["pytest"])
    observation = Observation(success=False, error="pytest falhou")

    strategist.capture_failure("objetivo", action, observation)
    created = strategist.flush()

    assert created
    seed = created[0]
    assert seed.priority == "high"
    assert "pytest" in seed.goal.lower()
    backlog_contents = (tmp_path / "backlog.yaml").read_text(encoding="utf-8")
    assert "llm_failure_capture" in backlog_contents
