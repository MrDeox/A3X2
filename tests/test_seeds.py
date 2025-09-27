from pathlib import Path

from a3x.seeds import SeedBacklog
from a3x.seed_runner import SeedRunner


def test_seed_backlog_load_and_pick(tmp_path: Path) -> None:
    backlog_file = tmp_path / "backlog.yaml"
    backlog_file.write_text(
        """
- id: one
  goal: A
  priority: high
- id: two
  goal: B
  priority: low
        """,
        encoding="utf-8",
    )
    backlog = SeedBacklog.load(backlog_file)
    first = backlog.next_seed()
    assert first is not None
    assert first.id == "one"


def test_seed_runner_executes_manual_seed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    script_path = tmp_path / "script.yaml"
    script_path.write_text(
        """
- type: write_file
  path: output.txt
  content: "seed runner"
- type: finish
  summary: "done"
        """,
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
llm:
  type: manual
  script: {script_path}
workspace:
  root: {workspace}
  allow_outside_root: false
limits:
  max_iterations: 4
  command_timeout: 30
  max_failures: 2
        """,
        encoding="utf-8",
    )

    backlog_file = tmp_path / "backlog.yaml"
    backlog_file.write_text(
        f"""
- id: seed.test
  goal: "Seed goal"
  priority: high
  config: {config_path}
        """,
        encoding="utf-8",
    )

    runner = SeedRunner(backlog_file)
    result = runner.run_next(default_config=config_path)

    assert result is not None
    assert result.completed is True
    assert (workspace / "output.txt").read_text(encoding="utf-8") == "seed runner"

    # backlog should mark as completed
    backlog = SeedBacklog.load(backlog_file)
    seed = backlog._seeds["seed.test"]
    assert seed.status == "completed"


def test_seed_requeue_with_backoff(tmp_path: Path) -> None:
    backlog_file = tmp_path / "backlog.yaml"
    backlog_file.write_text(
        """
- id: s1
  goal: do
  priority: high
  max_attempts: 2
        """,
        encoding="utf-8",
    )
    backlog = SeedBacklog.load(backlog_file)
    s = backlog.next_seed()
    assert s is not None
    backlog.mark_failed(s.id, notes="simulated failure")
    # Após falha, deve retornar a pending com next_run_at no futuro
    backlog = SeedBacklog.load(backlog_file)
    s2 = backlog._seeds["s1"]
    assert s2.status == "pending"
    assert s2.attempts == 1
    assert s2.next_run_at is not None
