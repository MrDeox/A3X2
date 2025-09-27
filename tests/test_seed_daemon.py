from pathlib import Path

from a3x.seed_daemon import run_loop


def test_seed_daemon_completes_all(tmp_path: Path) -> None:
    # Workspace and manual config/script
    workspace = tmp_path / "ws"
    workspace.mkdir()
    script = tmp_path / "script.yaml"
    script.write_text(
        """
- type: write_file
  path: out1.txt
  content: one
- type: finish
  summary: ok
        """,
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
llm:
  type: manual
  script: {script}
workspace:
  root: {workspace}
limits:
  max_iterations: 3
        """,
        encoding="utf-8",
    )

    backlog = tmp_path / "backlog.yaml"
    backlog.write_text(
        f"""
- id: a
  goal: do A
  priority: high
  config: {config}
- id: b
  goal: do B
  priority: low
  config: {config}
        """,
        encoding="utf-8",
    )

    res = run_loop(backlog=backlog, config=config, interval=0, stop_when_idle=True)
    assert res.runs == 2
    assert res.failed == 0
