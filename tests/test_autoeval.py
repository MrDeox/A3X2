from pathlib import Path

from a3x.autoeval import AutoEvaluator, EvaluationSeed
from a3x.capabilities import CapabilityRegistry
from a3x.planning.storage import load_mission_state


def test_auto_evaluator_records(tmp_path: Path, monkeypatch) -> None:
    class DummyGrowthGenerator:
        def __init__(self, history_path, output_path):
            self.history_path = history_path
            self.output_path = output_path
            DummyGrowthGenerator.instances.append(self)

        def ensure_tests(self) -> None:
            DummyGrowthGenerator.called = True

    DummyGrowthGenerator.instances = []
    DummyGrowthGenerator.called = False

    def dummy_report(**kwargs):
        dummy_report.called = True

    dummy_report.called = False

    monkeypatch.setattr("a3x.autoeval.GrowthTestGenerator", DummyGrowthGenerator)
    monkeypatch.setattr("a3x.autoeval.generate_capability_report", dummy_report)

    evaluator = AutoEvaluator(log_dir=tmp_path)
    evaluator.record(
        goal="Test",
        completed=True,
        iterations=3,
        failures=0,
        duration_seconds=1.23,
        seeds=[
            EvaluationSeed(
                description="Improve speed", priority="low", capability="core.diffing"
            )
        ],
        metrics={"actions_success_rate": 1.0, "iterations": 3},
        capabilities=["core.diffing"],
        human_feedback="fantastic",
    )

    log_file = tmp_path / "run_evaluations.jsonl"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8").strip()
    assert "Improve speed" in content
    assert "core.diffing" in content
    assert "actions_success_rate" in content
    assert DummyGrowthGenerator.called
    assert dummy_report.called

    summary = evaluator.latest_summary()
    assert "Última execução" in summary


def test_capability_maturity_updates(tmp_path: Path, monkeypatch) -> None:
    seed_dir = tmp_path / "seed"
    eval_dir = seed_dir / "evaluations"
    eval_dir.mkdir(parents=True)

    capabilities_yaml = seed_dir / "capabilities.yaml"
    capabilities_yaml.write_text(
        """
- id: core.diffing
  name: Diff
  category: vertical
  description: desc
  maturity: baseline
  metrics: {}
  seeds: []
- id: core.testing
  name: Testing
  category: vertical
  description: desc
  maturity: baseline
  metrics: {}
  seeds: []
        """,
        encoding="utf-8",
    )

    evaluator = AutoEvaluator(log_dir=eval_dir)

    metrics = {
        "core.diffing": {"success_rate": 0.96, "avg_iterations": 2.3},
        "core.testing": {"auto_trigger_rate": 0.8, "failures_detected": 0},
    }

    evaluator._update_capability_metrics(metrics)  # type: ignore[attr-defined]

    updated = CapabilityRegistry.from_yaml(capabilities_yaml)
    diff_cap = updated.get("core.diffing")
    testing_cap = updated.get("core.testing")

    assert diff_cap.maturity == "advanced"
    assert testing_cap.maturity == "established"


def test_mission_state_updates(tmp_path: Path, monkeypatch) -> None:
    seed_dir = tmp_path / "seed"
    eval_dir = seed_dir / "evaluations"
    eval_dir.mkdir(parents=True)

    (seed_dir / "capabilities.yaml").write_text(
        """
- id: core.diffing
  name: Diff
  category: vertical
  description: desc
  maturity: baseline
  metrics: {}
  seeds: []
- id: core.testing
  name: Testing
  category: vertical
  description: desc
  maturity: baseline
  metrics: {}
  seeds: []
        """,
        encoding="utf-8",
    )

    (seed_dir / "missions.yaml").write_text(
        """
missions:
  - id: core-diff
    vision: Diff mission
    status: active
    priority: high
    success_criteria: []
    target_metrics:
      core.diffing.success_rate:
        current: 0.0
        target: 0.95
        best: 0.0
        samples: 0
    capability_tags: [core.diffing]
    milestones:
      - id: diff-step
        goal: reach
        status: planned
        capability_tags: [core.diffing]
        metrics:
          core.diffing.success_rate:
            current: 0.0
            target: 0.9
            best: 0.0
            samples: 0
        notes: ""
    telemetry:
      metric_summaries: {}
      reflections: []
      discovered_gaps: []
      tools_required: []
      last_updated: null
version: 0.1
        """,
        encoding="utf-8",
    )

    evaluator = AutoEvaluator(log_dir=eval_dir)

    metrics = {
        "core.diffing": {
            "success_rate": 0.96,
            "avg_iterations": 2.0,
            "runs": 3,
            "completed_runs": 3,
        }
    }
    evaluator._update_capability_metrics(metrics)  # type: ignore[attr-defined]
    state_before = load_mission_state(seed_dir / "missions.yaml")
    evaluator._update_missions(metrics, state_before)  # type: ignore[attr-defined]

    state = load_mission_state(seed_dir / "missions.yaml")
    mission = state.find_mission("core-diff")
    assert mission is not None
    assert mission.status == "completed"
    assert mission.milestones[0].status == "completed"
