"""Auto-evaluation utilities for the SeedAI growth loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .testgen import GrowthTestGenerator
from .report import generate_capability_report
from .seeds import SeedBacklog, Seed
from .planner import Planner, PlannerThresholds


@dataclass
class EvaluationSeed:
    """Represents a proposed improvement generated from a run."""

    description: str
    priority: str = "medium"  # e.g., low/medium/high
    capability: Optional[str] = None  # link to capability id
    seed_type: str = "improvement"
    data: Optional[Dict[str, str]] = None


@dataclass
class RunEvaluation:
    """Snapshot of metrics collected after a run."""

    goal: str
    completed: bool
    iterations: int
    failures: int
    duration_seconds: Optional[float]
    timestamp: str
    seeds: List[EvaluationSeed]
    metrics: Dict[str, float]
    capabilities: List[str]
    human_feedback: Optional[str] = None
    notes: Optional[str] = None


class AutoEvaluator:
    """Persists evaluation entries for posteriores análises."""

    def __init__(
        self,
        log_dir: Path | str = Path("seed/evaluations"),
        thresholds: Optional[PlannerThresholds] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "run_evaluations.jsonl"
        self.base_dir = self.log_dir.parent
        self.metrics_dir = self.base_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history_file = self.metrics_dir / "history.json"
        self._growth_test_generator = GrowthTestGenerator(
            history_path=self.metrics_history_file,
            output_path=Path("tests/generated/test_metrics_growth.py"),
        )
        self.backlog_path = self.base_dir / "backlog.yaml"
        self.thresholds = thresholds or PlannerThresholds()

    def record(
        self,
        goal: str,
        completed: bool,
        iterations: int,
        failures: int,
        duration_seconds: Optional[float],
        seeds: Optional[List[EvaluationSeed]] = None,
        metrics: Optional[Dict[str, float]] = None,
        capabilities: Optional[List[str]] = None,
        human_feedback: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> RunEvaluation:
        evaluation = RunEvaluation(
            goal=goal,
            completed=completed,
            iterations=iterations,
            failures=failures,
            duration_seconds=duration_seconds,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seeds=seeds or [],
            metrics={k: float(v) for k, v in (metrics or {}).items()},
            capabilities=capabilities or [],
            human_feedback=human_feedback,
            notes=notes,
        )
        self._append(evaluation)
        return evaluation

    def _append(self, evaluation: RunEvaluation) -> None:
        entry = asdict(evaluation)
        entry["seeds"] = [asdict(seed) for seed in evaluation.seeds]
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._update_metric_history(evaluation.metrics)
        self._growth_test_generator.ensure_tests()
        generate_capability_report(
            metrics_history=self.metrics_history_file,
        )
        # Gera seeds automáticas se houver lacunas
        self._maybe_generate_auto_seeds()
        # Atualiza painel simples
        self._write_run_status()
        # Registrar reflexão pós-run
        self._write_reflection()

    def _update_metric_history(self, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        history: Dict[str, List[float]] = {}
        if self.metrics_history_file.exists():
            history = json.loads(self.metrics_history_file.read_text(encoding="utf-8"))
        for key, value in metrics.items():
            history.setdefault(key, []).append(float(value))
        with self.metrics_history_file.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)

    def latest_summary(self, max_metrics: int = 5) -> str:
        parts: List[str] = []
        last_eval = self._read_last_evaluation()
        if last_eval:
            status = "concluído" if last_eval.get("completed") else "não concluído"
            parts.append(
                f"Última execução: {status}, iterações={last_eval.get('iterations')}, falhas={last_eval.get('failures')}"
            )
            caps = last_eval.get("capabilities") or []
            if caps:
                parts.append("Capacidades exercitadas: " + ", ".join(caps))
        metrics_history = self._read_metrics_history()
        if metrics_history:
            summaries = []
            for name, values in list(metrics_history.items())[:max_metrics]:
                last = values[-1]
                best = max(values)
                summaries.append(f"{name} -> atual {last:.2f} (melhor {best:.2f})")
            parts.append("Métricas: " + "; ".join(summaries))
        return "\n".join(parts) if parts else "Sem histórico SeedAI registrado."

    def _read_last_evaluation(self) -> Optional[dict]:
        if not self.log_file.exists():
            return None
        with self.log_file.open("r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        if not lines:
            return None
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return None

    def _read_metrics_history(self) -> Dict[str, List[float]]:
        if not self.metrics_history_file.exists():
            return {}
        try:
            data = json.loads(self.metrics_history_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        history: Dict[str, List[float]] = {}
        for key, values in data.items():
            if not isinstance(values, list) or not values:
                continue
            history[key] = [float(v) for v in values[-50:]]  # limita tamanho
        return history

    # Painel simples -----------------------------------------------------------

    def _write_run_status(self) -> None:
        try:
            history = self._read_metrics_history()
            panel = self.base_dir / "reports" / "run_status.md"
            panel.parent.mkdir(parents=True, exist_ok=True)

            lines: List[str] = ["# SeedAI Run Status", "", "## Latest Metrics"]
            for metric in [
                "apply_patch_success_rate",
                "apply_patch_count",
                "actions_success_rate",
                "tests_success_rate",
                "lint_success_rate",
                "lint_run_count",
                "llm_latency_last",
            ]:
                vals = history.get(metric, [])
                if not vals:
                    continue
                lines.append(
                    f"- **{metric}**: last={vals[-1]:.4f} | best={max(vals):.4f} | samples={len(vals)}"
                )

            lines.append("")
            lines.append("## Recent Runs")
            eval_path = self.log_file
            entries: List[dict] = []
            if eval_path.exists():
                for line in eval_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        import json

                        entries.append(json.loads(line))
                    except Exception:
                        pass
            for entry in entries[-5:]:
                goal = entry.get("goal")
                completed = entry.get("completed")
                status = "✅" if completed else "⚠️"
                lines.append(
                    f"- {status} {entry.get('timestamp')}: {goal} (iter={entry.get('iterations')}, fail={entry.get('failures')})"
                )

            panel.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            # painel é auxiliar; não falhar hard
            pass

    def _write_reflection(self) -> None:
        try:
            last_eval = self._read_last_evaluation()
            history = self._read_metrics_history()
            reflection_path = self.base_dir / "reports" / "reflection.md"
            reflection_path.parent.mkdir(parents=True, exist_ok=True)

            lines: List[str] = ["# Run Reflection", ""]
            if last_eval:
                status = "✅ concluído" if last_eval.get("completed") else "⚠️ não concluído"
                lines.append(f"## Última Execução: {status}")
                lines.append(
                    f"- Objetivo: {last_eval.get('goal')}"
                )
                lines.append(
                    f"- Iterações: {last_eval.get('iterations')} | Falhas: {last_eval.get('failures')}"
                )
                duration = last_eval.get("duration_seconds")
                if duration is not None:
                    lines.append(f"- Duração: {duration:.2f}s")
                notes = last_eval.get("notes")
                if notes:
                    lines.append(f"- Notas: {notes}")
                seeds = last_eval.get("seeds") or []
                if seeds:
                    lines.append("")
                    lines.append("### Seeds sugeridas pelo run")
                    for seed in seeds:
                        lines.append(
                            f"- ({seed.get('seed_type')}) {seed.get('description')}"
                        )

            lines.append("")
            lines.append("## Métricas vs. Metas")
            thresholds = self.thresholds
            targets = {
                "apply_patch_success_rate": thresholds.apply_patch_success_rate,
                "actions_success_rate": thresholds.actions_success_rate,
                "tests_success_rate": thresholds.tests_success_rate,
                "lint_success_rate": thresholds.lint_success_rate,
            }
            for metric, threshold in targets.items():
                values = history.get(metric, [])
                if not values:
                    continue
                last = values[-1]
                status = "✅" if last >= threshold else "⚠️"
                lines.append(
                    f"- {status} {metric}: {last:.2f} (meta {threshold:.2f})"
                )

            reflection_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

    # Auto-seeds ---------------------------------------------------------------

    def _maybe_generate_auto_seeds(self) -> None:
        history = self._read_metrics_history()
        if not history:
            return
        backlog = SeedBacklog.load(self.backlog_path)

        seed_configs_root = (self.base_dir / "../configs").resolve()
        patch_config_path = str((seed_configs_root / "seed_patch.yaml").resolve())
        manual_config_path = str((seed_configs_root / "seed_manual.yaml").resolve())

        tests_config_path = str((seed_configs_root / "seed_tests.yaml").resolve())

        lint_config_path = str((seed_configs_root / "seed_lint.yaml").resolve())

        planner = Planner(thresholds=self.thresholds)
        for seed in planner.propose(
            history,
            patch_config_path=patch_config_path,
            manual_config_path=manual_config_path,
            tests_config_path=tests_config_path,
            lint_config_path=lint_config_path,
        ):
            if not backlog.exists(seed.id):
                backlog.add_seed(seed)


__all__ = [
    "AutoEvaluator",
    "EvaluationSeed",
    "RunEvaluation",
]
