from pathlib import Path

from a3x.report import generate_capability_report


def test_generate_capability_report(tmp_path: Path) -> None:
    capabilities = tmp_path / "capabilities.yaml"
    capabilities.write_text(
        """
- id: cap.one
  name: Capability One
  category: vertical
  description: Test
  maturity: baseline
        """,
        encoding="utf-8",
    )

    evaluations_dir = tmp_path / "evaluations"
    evaluations_dir.mkdir()
    log_file = evaluations_dir / "run_evaluations.jsonl"
    log_file.write_text(
        """
{"completed": true, "capabilities": ["cap.one"], "metrics": {}}
{"completed": false, "capabilities": ["cap.one"], "metrics": {}}
        """.strip(),
        encoding="utf-8",
    )

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "history.json").write_text(
        """
{"actions_success_rate": [0.5, 0.7]}
        """.strip(),
        encoding="utf-8",
    )

    output = tmp_path / "reports" / "capability_report.md"

    generate_capability_report(
        capabilities_path=capabilities,
        evaluations_log=log_file,
        metrics_history=metrics_dir / "history.json",
        output_path=output,
    )

    content = output.read_text(encoding="utf-8")
    assert "Capability One" in content
    assert "actions_success_rate" in content
    assert "Overall completion rate" in content
