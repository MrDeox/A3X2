from pathlib import Path

import pytest

from a3x.autoloop import GoalSpec, load_goal_rotation, run_autopilot


def test_load_goal_rotation(tmp_path: Path) -> None:
    rotation = tmp_path / "rotation.yaml"
    rotation.write_text(
        """
- goal: Test A
  config: configs/manual.yaml
  max_steps: 4
- goal: Test B
  config: configs/manual.yaml
        """,
        encoding="utf-8",
    )
    specs = load_goal_rotation(rotation)
    assert len(specs) == 2
    assert specs[0].goal == "Test A"
    assert specs[0].max_steps == 4


def test_run_autopilot_cycles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    goals = [GoalSpec(goal="Fake", config=tmp_path / "fake.yaml", max_steps=3)]

    class DummyResult:
        def __init__(self, completed: bool = True) -> None:
            self.completed = completed
            self.iterations = 1
            self.failures = 0

    calls = []

    from a3x import autoloop

    monkeypatch.setattr(
        autoloop, "_run_goal", lambda spec: calls.append(spec.goal) or DummyResult()
    )
    monkeypatch.setattr(autoloop, "_drain_seeds", lambda *args, **kwargs: None)

    exit_code = run_autopilot(
        goals,
        cycles=2,
        backlog_path=tmp_path / "seed" / "backlog.yaml",
        seed_default_config=tmp_path / "cfg.yaml",
        seed_max=None,
        seed_max_steps=None,
    )
    assert exit_code == 0
    assert calls == ["Fake", "Fake"]
    output = capsys.readouterr().out
    assert "Cycle 1" in output
    assert "Cycle 2" in output
