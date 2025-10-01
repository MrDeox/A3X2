from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from a3x.memory.insights import RetrospectiveReport
from a3x.policy import PolicyOverrideManager


class DummyAgent(SimpleNamespace):
    pass


def test_apply_and_update_overrides(tmp_path):
    manager = PolicyOverrideManager(path=tmp_path / "policy.yaml")
    agent = DummyAgent(recursion_depth=5, config=SimpleNamespace(limits=SimpleNamespace(max_failures=8)))

    manager.data = {"agent": {"recursion_depth": 6, "max_failures": 9}}
    manager.apply_to_agent(agent)
    assert agent.recursion_depth == 6
    assert agent.config.limits.max_failures == 9

    report = RetrospectiveReport(
        goal="",
        completed=False,
        iterations=3,
        failures=4,
        duration_seconds=None,
        metrics={},
        recommendations=["Reduzir profundidade recursiva", "aumentar supervisão"],
        notes=[],
    )
    manager.update_from_report(report, agent)
    assert agent.recursion_depth == 5  # 6 - 1
    assert agent.config.limits.max_failures == 8  # 9 - 1
    saved = (tmp_path / "policy.yaml").read_text(encoding="utf-8")
    assert "recursion_depth" in saved
