"""Auto-evaluation utilities for the SeedAI growth loop."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from .testgen import GrowthTestGenerator, E2ETestGenerator
from .report import generate_capability_report
from .seeds import SeedBacklog, Seed
from .planner import Planner, PlannerThresholds
from .planning.mission_planner import MissionPlanner
from .capabilities import CapabilityRegistry
from .capability_metrics import compute_capability_metrics
from .planning.storage import load_mission_state, save_mission_state
from .memory.store import SemanticMemory
from .memory.insights import build_insight_payload
from .config import AgentConfig
# Import MetaCapabilityPlanner locally to avoid circular import
# from .meta_capabilities import MetaCapabilityPlanner


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
    notes: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class AutoEvaluator:
    """Persists evaluation entries for posteriores análises."""

    def __init__(
        self,
        log_dir: Path | str = Path("seed/evaluations"),
        thresholds: Optional[PlannerThresholds] = None,
        config: Optional[AgentConfig] = None,
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
        self.capabilities_path = self.base_dir / "capabilities.yaml"
        self.missions_path = self.base_dir / "missions.yaml"
        self.curriculum_path = self.base_dir / "curriculum.yaml"
        self.curriculum_progress_path = self.base_dir / "reports" / "curriculum_progress.yaml"
        self.auto_critique_path = self.base_dir / "reports" / "auto_critique.md"
        self.config = config  # Store the config
        self.memory_path = self.base_dir / "memory" / "memory.jsonl"
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
        notes: Optional[str] = None,
        errors: Optional[List[str]] = None,
    ) -> RunEvaluation:
        # Calculate fitness_before (from previous run)
        fitness_before = self._calculate_fitness_before()
        
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
            notes=notes,
            errors=errors or [],
        )
        
        # Calculate fitness_after based on current metrics
        fitness_after = self._calculate_fitness(evaluation.metrics)
        
        # Calculate delta
        delta = fitness_after - fitness_before
        
        # Store the fitness data with the evaluation
        evaluation.metrics['fitness_before'] = fitness_before
        evaluation.metrics['fitness_after'] = fitness_after
        evaluation.metrics['fitness_delta'] = delta
        
        self._append(evaluation)
        return evaluation

    def _calculate_fitness_before(self) -> float:
        """Get the fitness from before this run."""
        # Load the last evaluation to get its fitness_after
        last_eval = self._read_last_evaluation()
        if last_eval and 'metrics' in last_eval:
            metrics = last_eval['metrics']
            if 'fitness_after' in metrics:
                return metrics['fitness_after']
        # If no previous fitness data, return default value
        return 0.0

    def _calculate_fitness(self, metrics: Dict[str, float]) -> float:
        """Calculate an overall fitness score based on metrics."""
        # Weighted combination of key metrics
        fitness = 0.0
        
        # Actions success rate (most important)
        fitness += metrics.get("actions_success_rate", 0.0) * 0.4
        
        # Apply patch success rate  
        if "apply_patch_success_rate" in metrics:
            fitness += metrics["apply_patch_success_rate"] * 0.3
        
        # Test success rate
        if "tests_success_rate" in metrics:
            fitness += metrics["tests_success_rate"] * 0.2
        
        # Recursion depth (efficiency) - capped at 1.0
        fitness += min(metrics.get("recursion_depth", 0) / 10.0, 1.0) * 0.1
        
        return fitness

    def _append(self, evaluation: RunEvaluation) -> None:
        entry = asdict(evaluation)
        entry["seeds"] = [asdict(seed) for seed in evaluation.seeds]
        entry["errors"] = evaluation.errors
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._update_metric_history(evaluation.metrics)
        
        # Analyze code quality and add to metrics
        code_quality_metrics = self._analyze_code_quality(evaluation)
        evaluation.metrics.update(code_quality_metrics)
        
        # Generate seeds based on code quality issues
        quality_seeds = self._check_code_quality_issues(code_quality_metrics)
        evaluation.seeds.extend(quality_seeds)

        # Calculate fitness delta for this run
        fitness_delta = evaluation.metrics.get('fitness_delta', 0.0)
        
        # Include fitness info in the metadata of seeds
        for seed in evaluation.seeds:
            if seed.data is None:
                seed.data = {}
            seed.data['fitness_delta'] = str(fitness_delta)
        
        # Enfileirar seeds originadas desta execução
        self.enqueue_seeds(evaluation.seeds, source="autoeval")
        
        self._growth_test_generator.ensure_tests()
        self._detect_and_run_e2e_tests(evaluation)
        generate_capability_report(
            metrics_history=self.metrics_history_file,
        )
        capability_metrics = compute_capability_metrics(self.log_file)
        mission_state = (
            load_mission_state(self.missions_path)
            if self.missions_path.exists()
            else None
        )
        registry = self._update_capability_metrics(capability_metrics)
        self._update_missions(capability_metrics, mission_state)
        # Gera seeds automáticas e meta-capabilities
        self._maybe_generate_auto_seeds(capability_metrics, mission_state, registry)
        # Armazena insight semântico
        self._update_semantic_memory(evaluation, capability_metrics)
        # Atualiza painel simples
        self._write_run_status()
        # Registrar reflexão pós-run
        self._write_reflection()
        # Auto-crítica e avanço de currículo
        self._post_reflection_follow_up()

    def _update_metric_history(self, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        history: Dict[str, List[float]] = {}
        if self.metrics_history_file.exists():
            try:
                history = json.loads(self.metrics_history_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                history = {} # Initialize as empty if corrupted
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
            # Adicionar resumo de erros
            errors = last_eval.get("errors")
            if errors:
                parts.append(f"Erros na última execução: {', '.join(errors[:2])}{'...' if len(errors) > 2 else ''}")
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
                status = (
                    "✅ concluído" if last_eval.get("completed") else "⚠️ não concluído"
                )
                lines.append(f"## Última Execução: {status}")
                lines.append(f"- Objetivo: {last_eval.get('goal')}")
                lines.append(
                    f"- Iterações: {last_eval.get('iterations')} | Falhas: {last_eval.get('failures')}"
                )
                duration = last_eval.get("duration_seconds")
                if duration is not None:
                    lines.append(f"- Duração: {duration:.2f}s")
                notes = last_eval.get("notes")
                if notes:
                    lines.append(f"- Notas: {notes}")
                errors = last_eval.get("errors") or []
                if errors:
                    lines.append("")
                    lines.append("### Erros da Execução")
                    for error in errors:
                        lines.append(f"- {error}")
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
                lines.append(f"- {status} {metric}: {last:.2f} (meta {threshold:.2f})")

            # Curriculum thresholds validation
            curriculum_thresholds = {
                "actions_success_rate": 0.95,
                "apply_patch_success_rate": 0.95,
                "recursion_depth": 5.0,
            }
            validation_status = self._validate_curriculum_thresholds(history)
            lines.append("")
            lines.append("## Validação de Curriculum Thresholds")
            lines.append(f"- Status Geral: {'✅ Passou' if validation_status else '❌ Falhou'}")
            for metric, target in curriculum_thresholds.items():
                values = history.get(metric, [])
                if values:
                    last = values[-1]
                    status = "✅" if (metric != "recursion_depth" and last >= target) or (metric == "recursion_depth" and last >= target) else "⚠️"
                    lines.append(f"- {status} {metric}: {last} (alvo {target})")

            reflection_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

    def _post_reflection_follow_up(self) -> None:
        """Gera auto-crítica e avança o currículo com base na última execução."""

        try:
            last_eval = self._read_last_evaluation()
            if not last_eval:
                return

            critique = self._build_auto_critique(last_eval)
            if critique:
                self._append_auto_critique_entry(last_eval, critique)

            self._progress_curriculum(last_eval)
        except Exception:
            # Não falhar o ciclo caso a etapa de auto-crítica tenha problemas
            pass

    def _build_auto_critique(self, evaluation: Dict[str, Any]) -> str:
        """Constrói texto curto de auto-crítica a partir dos resultados do run."""

        goal = evaluation.get("goal")
        completed = evaluation.get("completed")
        metrics = evaluation.get("metrics") or {}
        failures = evaluation.get("failures")
        iterations = evaluation.get("iterations")

        parts: List[str] = []
        if completed:
            parts.append("Execução concluída, avaliar próximos incrementos de qualidade.")
        else:
            parts.append("Objetivo não foi concluído; priorizar mitigação dos bloqueios identificados.")

        failure_rate = metrics.get("failure_rate")
        success_rate = metrics.get("success_rate")
        if isinstance(failure_rate, (int, float)) and failure_rate > 0.25:
            parts.append("Taxa de falhas acima de 25%; revisar estratégia de execução e cobertura de testes.")
        elif isinstance(success_rate, (int, float)) and success_rate < 0.8:
            parts.append("Sucesso abaixo de 80%; identificar gargalos antes de avançar no currículo.")

        if isinstance(iterations, int) and isinstance(failures, int) and failures > 0:
            parts.append(
                f"{failures} falhas em {iterations} iterações; documentar lições aprendidas no relatório."
            )

        errors = evaluation.get("errors") or []
        if errors:
            parts.append("Erros recorrentes detectados; tratar itens prioritários na próxima seed.")

        if not parts:
            return ""

        header = f"Auto-crítica do objetivo '{goal}'" if goal else "Auto-crítica"
        return f"{header}: {' '.join(parts)}"

    def _append_auto_critique_entry(self, evaluation: Dict[str, Any], critique: str) -> None:
        """Persiste auto-crítica incremental no relatório dedicado."""

        timestamp = datetime.now(timezone.utc).isoformat()
        lines: List[str] = []

        path = self.auto_critique_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            lines.append("# Auto-críticas do agente")
            lines.append("")

        goal = evaluation.get("goal") or "(sem objetivo registrado)"
        status = "✅" if evaluation.get("completed") else "⚠️"
        lines.append(f"## {timestamp} — {goal}")
        lines.append(f"Status: {status}")
        lines.append("")
        lines.append(critique)

        errors = evaluation.get("errors") or []
        if errors:
            lines.append("")
            lines.append("Principais erros capturados:")
            for error in errors[:3]:
                lines.append(f"- {error}")

        lines.append("")

        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _progress_curriculum(self, evaluation: Dict[str, Any]) -> None:
        steps = self._load_curriculum()
        if not steps:
            return

        progress = self._load_curriculum_progress()
        now = datetime.now(timezone.utc).isoformat()

        current_index = int(progress.get("current_index", 0))
        completed_steps: List[str] = list(progress.get("completed_steps", []))
        active_step_id: Optional[str] = progress.get("active_step_id")

        def find_step(step_id: Optional[str]) -> Optional[Dict[str, Any]]:
            if step_id is None:
                return None
            for step in steps:
                if str(step.get("id")) == str(step_id):
                    return step
            return None

        active_step = find_step(active_step_id)
        if not active_step and current_index < len(steps):
            active_step = steps[current_index]
            active_step_id = str(active_step.get("id"))

        goal = evaluation.get("goal")
        if active_step and goal == active_step.get("goal") and evaluation.get("completed"):
            step_id = str(active_step.get("id"))
            if step_id not in completed_steps:
                completed_steps.append(step_id)
            try:
                current_index = max(current_index, steps.index(active_step) + 1)
            except ValueError:
                current_index += 1
            active_step_id = None
            progress["last_completed_goal"] = goal
            progress["last_completed_at"] = now

        # Determina próximo passo e enfileira seed correspondente
        if current_index < len(steps):
            next_step = steps[current_index]
            next_step_id = str(next_step.get("id"))
            backlog_has_seed = self._has_curriculum_seed(next_step_id)

            if active_step_id is None and backlog_has_seed:
                active_step_id = next_step_id

            needs_enqueue = not backlog_has_seed
            if active_step_id not in (None, next_step_id):
                needs_enqueue = True

            if needs_enqueue:
                metadata = {
                    "curriculum_step_id": next_step_id,
                    "difficulty": next_step.get("difficulty", ""),
                    "title": next_step.get("title", ""),
                }
                priority_value = next_step.get("priority")
                if not priority_value:
                    priority_value = "medium"
                else:
                    priority_value = str(priority_value)
                seed = EvaluationSeed(
                    description=str(next_step.get("goal")),
                    priority=priority_value,
                    capability=None,
                    seed_type="curriculum",
                    data=metadata,
                )
                self.enqueue_seeds([seed], source="curriculum")
                active_step_id = next_step_id
                progress["last_enqueued_goal"] = next_step.get("goal")
                progress["last_enqueued_at"] = now

        progress["current_index"] = current_index
        ordered_ids = [str(step.get("id")) for step in steps]
        progress["completed_steps"] = [
            step_id for step_id in ordered_ids if step_id in completed_steps
        ]
        progress["active_step_id"] = active_step_id
        progress["last_goal_evaluated"] = goal
        progress["updated_at"] = now

        self._save_curriculum_progress(progress)

    def _has_curriculum_seed(self, step_id: str) -> bool:
        backlog = SeedBacklog.load(self.backlog_path)
        slug = self._slugify(str(step_id))
        target_prefix = f"curriculum.{slug}"
        for seed_id in backlog.list_all_ids():
            if seed_id.startswith(target_prefix):
                return True
        return False

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "seed"

    def _load_curriculum(self) -> List[Dict[str, Any]]:
        if not self.curriculum_path.exists():
            return []
        try:
            data = yaml.safe_load(self.curriculum_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return []
        steps = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(steps, list):
            return []
        return [step for step in steps if isinstance(step, dict)]

    def _load_curriculum_progress(self) -> Dict[str, Any]:
        path = self.curriculum_progress_path
        if not path.exists():
            return {"current_index": 0, "completed_steps": []}
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                return {"current_index": 0, "completed_steps": []}
            return data
        except Exception:
            return {"current_index": 0, "completed_steps": []}

    def _save_curriculum_progress(self, progress: Dict[str, Any]) -> None:
        path = self.curriculum_progress_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(progress, fh, allow_unicode=True, sort_keys=False)

    def _validate_curriculum_thresholds(self, history: Dict[str, List[float]]) -> bool:
        """Validate post-run metrics against curriculum thresholds."""
        thresholds = {
            "actions_success_rate": 0.95,
            "apply_patch_success_rate": 0.95,
        }
        for metric, target in thresholds.items():
            values = history.get(metric, [])
            if values and values[-1] < target:
                return False
        # Check recursion_depth >=5
        recursion_values = history.get("recursion_depth", [])
        if recursion_values and recursion_values[-1] < 5:
            return False
        return True
    # Auto-seeds ---------------------------------------------------------------

    def _maybe_generate_auto_seeds(
        self,
        capability_metrics: Dict[str, Dict[str, float | int | None]],
        mission_state,
        registry: CapabilityRegistry | None,
    ) -> None:
        history = self._read_metrics_history()
        if not history:
            return
        backlog = SeedBacklog.load(self.backlog_path)

        seed_configs_root = (self.base_dir / "../configs").resolve()
        patch_config_path = str((seed_configs_root / "seed_patch.yaml").resolve())
        manual_config_path = str((seed_configs_root / "seed_manual.yaml").resolve())

        tests_config_path = str((seed_configs_root / "seed_tests.yaml").resolve())

        lint_config_path = str((seed_configs_root / "seed_lint.yaml").resolve())

        last_eval = self._read_last_evaluation()
        last_errors = last_eval.get("errors") if last_eval else None

        planner = Planner(thresholds=self.thresholds)
        planner_seeds = []
        for seed in planner.propose(
            history,
            patch_config_path=patch_config_path,
            manual_config_path=manual_config_path,
            tests_config_path=tests_config_path,
            lint_config_path=lint_config_path,
            capability_metrics=capability_metrics,
            last_errors=last_errors,
        ):
            planner_seeds.append(seed)

        for seed in planner_seeds:
            processed_metadata = {
                k: ", ".join(v) if isinstance(v, list) else str(v)
                for k, v in seed.metadata.items()
            }
            processed_seed = Seed(
                id=seed.id,
                goal=seed.goal,
                priority=seed.priority,
                status=seed.status,
                type=seed.type,
                config=seed.config,
                max_steps=seed.max_steps,
                metadata=processed_metadata,
                history=seed.history,
                attempts=seed.attempts,
                max_attempts=seed.max_attempts,
                next_run_at=seed.next_run_at,
                last_error=seed.last_error,
            )
            if not backlog.exists(processed_seed.id):
                backlog.add_seed(processed_seed)

        if mission_state:
            mission_planner = MissionPlanner()
            mission_seeds = []
            for seed in mission_planner.propose(
                mission_state,
                capability_metrics,
                patch_config_path=patch_config_path,
                manual_config_path=manual_config_path,
                tests_config_path=tests_config_path,
                lint_config_path=lint_config_path,
            ):
                mission_seeds.append(seed)

            for seed in mission_seeds:
                processed_metadata = {
                    k: ", ".join(v) if isinstance(v, list) else str(v)
                    for k, v in seed.metadata.items()
                }
                processed_seed = Seed(
                    id=seed.id,
                    goal=seed.goal,
                    priority=seed.priority,
                    status=seed.status,
                    type=seed.type,
                    config=seed.config,
                    max_steps=seed.max_steps,
                    metadata=processed_metadata,
                    history=seed.history,
                    attempts=seed.attempts,
                    max_attempts=seed.max_attempts,
                    next_run_at=seed.next_run_at,
                    last_error=seed.last_error,
                )
                if not backlog.exists(processed_seed.id):
                    backlog.add_seed(processed_seed)

        if registry:
            # Import MetaCapabilityEngine locally to avoid circular import
            from .meta_capabilities import MetaCapabilityEngine

            meta_engine = MetaCapabilityEngine(config=self.config, auto_evaluator=self)
            proposals = meta_engine.propose_new_skills()
            meta_seeds: List[EvaluationSeed] = []
            for proposal in proposals:
                feasible, score, reason = meta_engine.evaluate_proposal_feasibility(proposal)
                if not feasible:
                    continue
                seed = meta_engine.create_skill_seed(proposal)
                data = dict(seed.data or {})
                data.setdefault("feasibility_score", f"{score:.2f}")
                data.setdefault("feasibility_reason", reason)
                seed.data = data
                meta_seeds.append(seed)
                meta_engine.save_skill_proposal(proposal)

            if meta_seeds:
                self.enqueue_seeds(meta_seeds, source="meta_capabilities")

    def enqueue_seeds(
        self, seeds: List[EvaluationSeed], *, source: str = "autoeval"
    ) -> None:
        """Add evaluation seeds to the shared backlog."""

        if not seeds:
            return

        backlog = SeedBacklog.load(self.backlog_path)
        timestamp = datetime.now(timezone.utc)
        created_at = timestamp.isoformat()
        default_config = self._resolve_default_seed_config()

        for index, eval_seed in enumerate(seeds):
            slug_source = (
                (eval_seed.data or {}).get("proposal_id")
                or (eval_seed.data or {}).get("seed_id")
                or eval_seed.description
            )
            slug = re.sub(r"[^a-z0-9]+", "-", slug_source.lower()).strip("-")
            if not slug:
                slug = "seed"
            seed_id = f"{source}.{slug}"
            if backlog.exists(seed_id):
                seed_id = f"{seed_id}-{timestamp.strftime('%H%M%S')}-{index}"

            metadata = {
                "source": source,
                "created_at": created_at,
                "seed_type": eval_seed.seed_type,
            }
            if eval_seed.capability:
                metadata["capability"] = eval_seed.capability
            for key, value in (eval_seed.data or {}).items():
                if key == "generated_code" and isinstance(value, str):
                    metadata["data.generated_code_length"] = str(len(value))
                else:
                    metadata[f"data.{key}"] = str(value)

            backlog.add_seed(
                Seed(
                    id=seed_id,
                    goal=eval_seed.description,
                    priority=eval_seed.priority or "medium",
                    status="pending",
                    type=eval_seed.seed_type or "improvement",
                    config=default_config,
                    metadata=metadata,
                )
            )

    def _resolve_default_seed_config(self) -> str:
        if self.config and self.config.loop.seed_config:
            return str(self.config.loop.seed_config)
        configs_dir = (self.base_dir / "../configs").resolve()
        manual_config = configs_dir / "seed_manual.yaml"
        return str(manual_config)

    def _update_capability_metrics(
        self, capability_metrics: Dict[str, Dict[str, float | int | None]]
    ) -> CapabilityRegistry | None:
        if not self.capabilities_path.exists() or not capability_metrics:
            return None
        registry = CapabilityRegistry.from_yaml(self.capabilities_path)
        registry.update_metrics(capability_metrics)
        maturity_updates = self._derive_maturity_updates(registry, capability_metrics)
        if maturity_updates:
            registry.update_maturity(maturity_updates)
        header = None
        try:
            first_line = self.capabilities_path.read_text(
                encoding="utf-8"
            ).splitlines()[0]
            if first_line.startswith("#"):
                header = first_line
        except Exception:
            header = None
        registry.to_yaml(self.capabilities_path, header_comment=header)
        return registry

    def _derive_maturity_updates(
        self,
        registry: CapabilityRegistry,
        capability_metrics: Dict[str, Dict[str, float | int | None]],
    ) -> Dict[str, str]:
        updates: Dict[str, str] = {}

        diff = capability_metrics.get("core.diffing", {})
        success_rate = diff.get("success_rate")
        if isinstance(success_rate, (int, float)):
            if success_rate >= 0.95:
                updates["core.diffing"] = "advanced"
            elif success_rate >= 0.9:
                updates["core.diffing"] = "established"
            else:
                updates["core.diffing"] = "baseline"

        testing = capability_metrics.get("core.testing", {})
        auto_rate = testing.get("auto_trigger_rate")
        failures = testing.get("failures_detected")
        if isinstance(auto_rate, (int, float)):
            if auto_rate >= 0.85 and not failures:
                updates["core.testing"] = "advanced"
            elif auto_rate >= 0.7 and (not failures or failures == 0):
                updates["core.testing"] = "established"
            else:
                updates["core.testing"] = "baseline"

        python_cap = capability_metrics.get("horiz.python", {})
        tasks_completed = python_cap.get("tasks_completed")
        regression_rate = python_cap.get("regression_rate")
        if isinstance(tasks_completed, (int, float)):
            if tasks_completed >= 25 and (regression_rate in (0, 0.0, None)):
                updates["horiz.python"] = "advanced"
            elif tasks_completed >= 10 and (
                regression_rate is None or regression_rate <= 0.05
            ):
                updates["horiz.python"] = "established"
            else:
                updates["horiz.python"] = "baseline"

        docs_cap = capability_metrics.get("horiz.docs", {})
        docs_generated = docs_cap.get("docs_generated")
        if isinstance(docs_generated, (int, float)):
            if docs_generated >= 20:
                updates["horiz.docs"] = "advanced"
            elif docs_generated >= 8:
                updates["horiz.docs"] = "established"
            else:
                updates["horiz.docs"] = "baseline"

        # preserve existing maturity when no change
        final_updates: Dict[str, str] = {}
        for cap_id, maturity in updates.items():
            try:
                current = registry.get(cap_id).maturity
            except KeyError:
                continue
            if current != maturity:
                final_updates[cap_id] = maturity
        return final_updates

    def _update_missions(
        self,
        capability_metrics: Dict[str, Dict[str, float | int | None]],
        mission_state,
    ) -> None:
        if mission_state is None or not capability_metrics:
            return
        state = mission_state
        changed = False

        def resolve(metric_ref: str) -> float | None:
            if not metric_ref:
                return None
            if "." in metric_ref:
                capability, metric = metric_ref.rsplit(".", 1)
            else:
                capability, metric = metric_ref, ""
            data = capability_metrics.get(capability)
            if not data:
                return None
            if metric:
                value = data.get(metric)
            else:
                value = None
            if isinstance(value, (int, float)):
                return float(value)
            return None

        for mission in state.missions:
            # Atualiza métricas-alvo de missão
            for metric_name, snapshot in mission.target_metrics.items():
                value = resolve(metric_name)
                if value is None:
                    continue
                if snapshot.current != value:
                    snapshot.current = value
                    changed = True
                snapshot.samples = int(snapshot.samples or 0) + 1
                snapshot.best = (
                    value if snapshot.best is None else max(snapshot.best, value)
                )

            # Atualiza milestones
            for milestone in mission.milestones:
                all_met = True
                some_progress = False
                for metric_name, snapshot in milestone.metrics.items():
                    value = resolve(metric_name)
                    if value is None:
                        all_met = False
                        continue
                    if snapshot.current != value:
                        snapshot.current = value
                        changed = True
                    snapshot.samples = int(snapshot.samples or 0) + 1
                    snapshot.best = (
                        value if snapshot.best is None else max(snapshot.best, value)
                    )
                    target = snapshot.target
                    if target is not None and value < target:
                        all_met = False
                    if value > 0:
                        some_progress = True
                prev_status = milestone.status
                if all_met and milestone.metrics:
                    milestone.status = "completed"
                elif some_progress:
                    milestone.status = "in_progress"
                else:
                    milestone.status = milestone.status or "planned"
                if milestone.status != prev_status:
                    changed = True

            prev_status = mission.status
            if mission.milestones and all(
                m.status == "completed" for m in mission.milestones
            ):
                mission.status = "completed"
            elif any(
                m.status in {"in_progress", "completed"} for m in mission.milestones
            ):
                mission.status = "active"
            else:
                mission.status = mission.status or "draft"
            if mission.status != prev_status:
                changed = True

            # Telemetria
            metric_items = []
            for name, snapshot in mission.target_metrics.items():
                metric_items.append((name, snapshot))
            mission.telemetry.merge_metrics(metric_items)

        if changed:
            save_mission_state(state, self.missions_path)

    def _analyze_code_quality(self, evaluation: RunEvaluation) -> Dict[str, float]:
        """Analyze code quality metrics from the evaluation."""
        # Calculate code quality metrics based on the actions and metrics in the evaluation
        quality_metrics = {}
        
        # Look for write file or patch apply actions in metrics
        if 'apply_patch_count' in evaluation.metrics:
            patch_count = evaluation.metrics['apply_patch_count']
            if patch_count > 0:
                # For now just record the count, but we can expand to analyze patch content
                quality_metrics['apply_patch_count'] = float(patch_count)
        
        # Look for any code-related metrics that could indicate quality
        if 'unique_file_extensions' in evaluation.metrics:
            extensions = evaluation.metrics['unique_file_extensions']
            # Value Python files higher for quality analysis
            if extensions > 0:
                quality_metrics['file_diversity'] = float(extensions)
        
        # Analyze actual code changes if possible by looking at the history
        # This would require access to the actual patch/diff content which normally comes from the history
        # For now, we'll add a placeholder and enhance it later
        
        # Add more quality metrics based on what we can infer from the evaluation
        # Calculate failure rate as an indicator of "quality" of implementation
        if evaluation.iterations > 0:
            failure_rate = evaluation.failures / evaluation.iterations if evaluation.iterations > 0 else 0.0
            quality_metrics['failure_rate'] = float(failure_rate)
            quality_metrics['success_rate'] = 1.0 - failure_rate
            
        return quality_metrics

    def analyze_code_complexity_from_patch(self, patch_content: str) -> Dict[str, float]:
        """Analyze code complexity from a patch/diff content."""
        complexity_metrics = {}
        
        # Extract Python code changes from the patch
        python_changes = self._extract_python_code_from_patch(patch_content)
        
        if python_changes:
            # Analyze the Python code for complexity
            try:
                tree = ast.parse(python_changes)
                complexity_info = self._analyze_ast_complexity(tree)
                complexity_metrics.update(complexity_info)
            except SyntaxError:
                # If parsing fails, skip complexity analysis for this patch
                pass
        
        return complexity_metrics

    def _extract_python_code_from_patch(self, patch_content: str) -> str:
        """Extract actual Python code from patch content."""
        lines = patch_content.split('\n')
        python_code = []
        
        in_diff = False
        for line in lines:
            if line.startswith('+++ ') and line.endswith('.py'):
                in_diff = True
                continue
            elif line.startswith('--- '):
                in_diff = False
                continue
            
            if in_diff and line.startswith('+'):
                # This is a line being added
                code_line = line[1:]  # Remove the '+' prefix
                python_code.append(code_line)
            elif in_diff and line.startswith(' '):
                # This is a context line
                code_line = line[1:]  # Remove the space prefix
                python_code.append(code_line)
        
        return '\n'.join(python_code)

    def _analyze_ast_complexity(self, tree: ast.AST) -> Dict[str, float]:
        """Analyze AST for complexity metrics."""
        stats = {
            'function_count': 0,
            'class_count': 0,
            'total_nodes': 0,
            'max_depth': 0,
        }
        
        def count_nodes(node, depth=0):
            stats['total_nodes'] += 1
            stats['max_depth'] = max(stats['max_depth'], depth)
            
            if isinstance(node, ast.FunctionDef):
                stats['function_count'] += 1
            elif isinstance(node, ast.ClassDef):
                stats['class_count'] += 1
            
            for child in ast.iter_child_nodes(node):
                count_nodes(child, depth + 1)
        
        for node in ast.iter_child_nodes(tree):
            count_nodes(node)
        
        # Convert counts to floats for metrics
        complexity_metrics = {
            'ast_function_count': float(stats['function_count']),
            'ast_class_count': float(stats['class_count']),
            'ast_total_nodes': float(stats['total_nodes']),
            'ast_max_depth': float(stats['max_depth']),
        }
        
        return complexity_metrics

    def _update_semantic_memory(
        self,
        evaluation: RunEvaluation,
        capability_metrics: Dict[str, Dict[str, float | int | None]],
    ) -> None:
        try:
            store = SemanticMemory(self.memory_path)
            title, content, tags, metadata = build_insight_payload(
                evaluation, capability_metrics
            )
            # Avoid duplicates by basic check (goal + timestamp)
            key = {"goal": evaluation.goal, "timestamp": evaluation.timestamp}
            if any(
                entry.metadata.get("goal") == key["goal"]
                and entry.metadata.get("timestamp") == key["timestamp"]
                for entry in store.entries
            ):
                return
            store.add(title, content, tags=tags, metadata=metadata)
        except RuntimeError:
            # sentence-transformers not installed; skip silently
            return
        except Exception:
            return

    def _check_code_quality_issues(self, quality_metrics: Dict[str, float]) -> List[EvaluationSeed]:
        """Generate seeds based on code quality issues."""
        seeds = []
        
        # Check if there are too many failures (indicating poor implementation quality)
        if quality_metrics.get('failure_rate', 0) > 0.3:  # More than 30% failure rate
            seeds.append(
                EvaluationSeed(
                    description="Reduzir taxa de falhas durante execução (alta taxa de falhas detectada).",
                    priority="high",
                    capability="core.execution",
                    seed_type="quality",
                    data={"metric": "failure_rate", "value": str(quality_metrics.get('failure_rate', 0))}
                )
            )
        
        # Check if too many patches are being applied without proper success
        if (quality_metrics.get('apply_patch_count', 0) > 5 and 
            quality_metrics.get('success_rate', 1) < 0.7):
            seeds.append(
                EvaluationSeed(
                    description="Melhorar qualidade das alterações de código aplicadas (muitos patches com baixa taxa de sucesso).",
                    priority="medium",
                    capability="core.diffing",
                    seed_type="quality",
                    data={"metric": "apply_patch_success_rate", "value": str(quality_metrics.get('success_rate', 1))}
                )
            )
        
        # Check if the system is not diversifying file types enough (might indicate lack of features)
        if quality_metrics.get('file_diversity', 0) < 2 and quality_metrics.get('apply_patch_count', 0) > 10:
            seeds.append(
                EvaluationSeed(
                    description="Expandir diversidade de tipos de arquivos manipulados (sistema focado em poucos tipos de arquivos).",
                    priority="low",
                    capability="horiz.file_handling",
                    seed_type="quality",
                    data={"metric": "file_diversity", "value": str(quality_metrics.get('file_diversity', 0))}
                )
            )
            
        return seeds


    def _detect_code_modifications(self) -> bool:
        """Detect if any Python files in a3x/ have been modified since last check."""
        check_file = self.log_dir / "last_mod_check.txt"
        last_time = 0.0
        if check_file.exists():
            try:
                last_time = datetime.fromisoformat(check_file.read_text(encoding="utf-8")).timestamp()
            except (ValueError, OSError):
                pass
        py_files = list(Path("a3x").rglob("*.py"))
        if not py_files:
            return False
        try:
            current_max_mtime = max(f.stat().st_mtime for f in py_files)
            if current_max_mtime > last_time:
                check_file.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
                return True
        except OSError:
            pass
        return False

    def _detect_and_run_e2e_tests(self, evaluation: RunEvaluation) -> None:
        """Detect code mods, generate/ensure e2e tests, run them, seed failures if any."""
        if not self._detect_code_modifications():
            return
        try:
            gen = E2ETestGenerator()
            gen.generate_basic_cycle_test()
            gen.generate_multi_cycle_test(cycles=3)
            gen.generate_seed_runner_test()
            result = subprocess.run(
                ["pytest", "-q", "tests/integration/"],
                capture_output=True,
                text=True,
                cwd=str(Path.cwd())
            )
            if result.returncode != 0:
                failure_seed = EvaluationSeed(
                    description="Falha em testes E2E após modificações de código - revisar integrações e rollbacks de segurança.",
                    priority="high",
                    capability="core.testing",
                    seed_type="e2e_failure",
                    data={"pytest_stderr": result.stderr[:1000] if result.stderr else None}
                )
                evaluation.seeds.append(failure_seed)
                # Seed back to backlog
                backlog = SeedBacklog.load(self.backlog_path)
                seed_id = (
                    f"e2e_failure_{datetime.now(timezone.utc).isoformat().replace(':', '-').split('.')[0].replace('+00:00', 'Z')}"
                )
                seed_obj = Seed(
                    id=seed_id,
                    goal="Investigar e corrigir falha detectada nos testes E2E automáticos.",
                    priority=failure_seed.priority,
                    type=failure_seed.seed_type,
                    metadata={
                        "source": "autoeval",
                        "capability": failure_seed.capability or "",
                    },
                )
                backlog.add_seed(seed_obj)
        except Exception:
            # Non-critical; log if needed but don't fail eval
            pass


__all__ = [
    "AutoEvaluator",
    "EvaluationSeed",
    "RunEvaluation",
]
