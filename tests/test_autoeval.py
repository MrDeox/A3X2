from pathlib import Path

from a3x.autoeval import AutoEvaluator, EvaluationSeed


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
        seeds=[EvaluationSeed(description="Improve speed", priority="low", capability="core.diffing")],
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
