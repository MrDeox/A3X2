from pathlib import Path

import pytest

from a3x.cli import main as cli_main


def test_cli_run_with_autowatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prepare manual script and config
    ws = tmp_path / "ws"
    ws.mkdir()
    script_run = tmp_path / "script_run.yaml"
    script_run.write_text(
        """
- type: message
  text: ok
- type: finish
  summary: done
        """,
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
llm:
  type: manual
  script: {script_run}
workspace:
  root: {ws}
limits:
  max_iterations: 3
        """,
        encoding="utf-8",
    )

    # Backlog with one seed that writes a file
    script_seed = tmp_path / "script_seed.yaml"
    script_seed.write_text(
        """
- type: write_file
  path: created.txt
  content: hello
- type: finish
  summary: ok
        """,
        encoding="utf-8",
    )
    seed_config = tmp_path / "seed_config.yaml"
    seed_config.write_text(
        f"""
llm:
  type: manual
  script: {script_seed}
workspace:
  root: {ws}
limits:
  max_iterations: 3
        """,
        encoding="utf-8",
    )
    backlog = tmp_path / "backlog.yaml"
    backlog.write_text(
        f"""
- id: only
  goal: create
  config: {seed_config}
        """,
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    # Run CLI com auto-watch apontando para backlog/config temporários
    code = cli_main(
        [
            "run",
            "--goal",
            "noop",
            "--config",
            str(config),
            "--auto-watch",
            "--watch-backlog",
            str(backlog),
            "--watch-interval",
            "0",
            "--watch-max-runs",
            "1",
        ]
    )
    assert code == 0
    assert (ws / "created.txt").exists()


def test_cli_run_with_loop_autoseed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()

    script_run = tmp_path / "script_run.yaml"
    script_run.write_text(
        """
- type: message
  text: run
- type: finish
  summary: done
        """,
        encoding="utf-8",
    )

    seed_script = tmp_path / "seed_script.yaml"
    seed_script.write_text(
        """
- type: write_file
  path: created.txt
  content: loop
- type: finish
  summary: ok
        """,
        encoding="utf-8",
    )

    seed_config = tmp_path / "seed_config.yaml"
    seed_config.write_text(
        f"""
llm:
  type: manual
  script: {seed_script}
workspace:
  root: {ws}
limits:
  max_iterations: 3
        """,
        encoding="utf-8",
    )

    backlog = tmp_path / "backlog.yaml"
    backlog.write_text(
        """
- id: only
  goal: create
  priority: high
        """,
        encoding="utf-8",
    )

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
llm:
  type: manual
  script: {script_run}
workspace:
  root: {ws}
limits:
  max_iterations: 3
loop:
  auto_seed: true
  seed_backlog: {backlog}
  seed_config: {seed_config}
  seed_interval: 0
  stop_when_idle: true
        """,
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    code = cli_main(
        [
            "run",
            "--goal",
            "noop",
            "--config",
            str(config),
        ]
    )

    assert code == 0
    assert (ws / "created.txt").exists()
