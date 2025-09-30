"""Auto-evaluation utilities for the SeedAI growth loop."""

from __future__ import annotations

import ast
import json
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .testgen import GrowthTestGenerator
from .report import generate_capability_report
from .seeds import SeedBacklog
from .planner import Planner, PlannerThresholds
from .planning.mission_planner import MissionPlanner
from .capabilities import CapabilityRegistry
from .capability_metrics import compute_capability_metrics
from .planning.storage import load_mission_state, save_mission_state
from .memory.store import SemanticMemory
from .memory.insights import build_insight_payload
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
    human_feedback: Optional[str] = None
    notes: Optional[str] = None


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
        
        # Analyze code quality and add to metrics
        code_quality_metrics = self._analyze_code_quality(evaluation)
        evaluation.metrics.update(code_quality_metrics)
        
        # Generate seeds based on code quality issues
        quality_seeds = self._check_code_quality_issues(code_quality_metrics)
        evaluation.seeds.extend(quality_seeds)
        
        self._growth_test_generator.ensure_tests()
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

        planner = Planner(thresholds=self.thresholds)
        for seed in planner.propose(
            history,
            patch_config_path=patch_config_path,
            manual_config_path=manual_config_path,
            tests_config_path=tests_config_path,
            lint_config_path=lint_config_path,
            capability_metrics=capability_metrics,
        ):
            if not backlog.exists(seed.id):
                backlog.add_seed(seed)

        if mission_state:
            mission_planner = MissionPlanner()
            for seed in mission_planner.propose(
                mission_state,
                capability_metrics,
                patch_config_path=patch_config_path,
                manual_config_path=manual_config_path,
                tests_config_path=tests_config_path,
                lint_config_path=lint_config_path,
            ):
                if not backlog.exists(seed.id):
                    backlog.add_seed(seed)

        if registry:
            config_map = {
                "patch": patch_config_path,
                "tests": tests_config_path,
                "lint": lint_config_path,
                "manual": manual_config_path,
            }
            # Import MetaCapabilityEngine locally to avoid circular import
            from .meta_capabilities import MetaCapabilityEngine
            meta_engine = MetaCapabilityEngine(config=self.config, auto_evaluator=self)
            existing_ids = backlog.list_all_ids()
            for seed in meta_engine.propose_new_skills():
                if not backlog.exists(seed.id):
                    backlog.add_seed(seed)

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


__all__ = [
    "AutoEvaluator",
    "EvaluationSeed",
    "RunEvaluation",
]
