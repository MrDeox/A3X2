from pathlib import Path

from a3x.autoeval import AutoEvaluator
from a3x.seeds import SeedBacklog


def _setup_seed_configs(root: Path) -> None:
    configs_dir = root / "configs"
    scripts_dir = configs_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "patch_doc.yaml").write_text(
        """
- type: write_file
  path: configs/doc.md
  content: "# Doc\n"
- type: patch
  diff: |
    --- a/configs/doc.md
    +++ b/configs/doc.md
    @@
    -# Doc
    +# Doc
    +Linha
- type: finish
  summary: ok
        """,
        encoding="utf-8",
    )
    (configs_dir / "seed_patch.yaml").write_text(
        """
llm:
  type: manual
  script: configs/scripts/patch_doc.yaml
workspace:
  root: .
        """,
        encoding="utf-8",
    )
    (configs_dir / "seed_manual.yaml").write_text(
        """
llm:
  type: manual
  script: configs/scripts/patch_doc.yaml
workspace:
  root: .
        """,
        encoding="utf-8",
    )


def test_auto_seeds_created(tmp_path: Path) -> None:
    _setup_seed_configs(tmp_path)
    seed_dir = tmp_path / "seed"
    eval_dir = seed_dir / "evaluations"
    metrics_dir = seed_dir / "metrics"
    eval_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    # 1) nenhum patch ainda -> cria benchmark diff/report
    (metrics_dir / "history.json").write_text(
        '{"apply_patch_count": [0, 0], "actions_total": [1, 2]}', encoding="utf-8"
    )

    evaluator = AutoEvaluator(log_dir=eval_dir)
    evaluator.record(
        goal="noop",
        completed=True,
        iterations=1,
        failures=0,
        duration_seconds=0.1,
        metrics={"actions_total": 2.0},
    )

    backlog = SeedBacklog.load(seed_dir / "backlog.yaml")
    assert backlog.exists("auto.benchmark.diff")
    assert backlog.exists("auto.benchmark.report")

    # 2) patches executados mas taxa de sucesso baixa -> auto.patch.success
    (metrics_dir / "history.json").write_text(
        '{"apply_patch_count": [0, 1], "apply_patch_success_rate": [0.0]}',
        encoding="utf-8",
    )
    evaluator.record(
        goal="patch",
        completed=True,
        iterations=2,
        failures=1,
        duration_seconds=0.2,
        metrics={"apply_patch_count": 1.0, "apply_patch_success_rate": 0.0},
    )

    backlog = SeedBacklog.load(seed_dir / "backlog.yaml")
    assert backlog.exists("auto.patch.success")
