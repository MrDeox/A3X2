"""Simple planner that proposes auto-seeds based on metric gaps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import json
import yaml
from .llm import OpenRouterLLMClient
from .actions import AgentState
from .memory.insights import StatefulRetriever



@dataclass
class PlannerThresholds:
    apply_patch_success_rate: float = 0.8
    actions_success_rate: float = 0.8
    tests_success_rate: float = 0.9
    lint_success_rate: float = 0.9


@dataclass
class PromptTemplate:
    base: str = """Propose seeds to improve the A3X agent based on current metrics and history."""
    low_rate: str = """The actions_success_rate is {rate:.2f}, which is below the target of 0.9. Prioritize seeds that enhance action selection, planning, and execution. Consider self-modification for tuning prompts in planner.py or agent.py to enable better recursion and reduce failures."""
    examples: str = """Examples of high-ROI actions and seeds:

High-ROI for low patch success:
- id: auto.benchmark.diff
  goal: "Apply a simple unified diff to update a documentation file."
  priority: high
  type: benchmark_diff
  config: configs/scripts/seed_patch.yaml
  max_steps: 5
  metadata:
    description: "Benchmark seed to practice and improve diff application success."

For self-modify to tune planning:
- id: meta.planner_tune
  goal: "Enhance prompt templates in a3x/planner.py with chain-of-thought reasoning and examples for high-ROI action selection."
  priority: high
  type: refactor
  config: configs/scripts/seed_patch.yaml
  max_steps: 8
  metadata:
    description: "Refactor to dynamically refine prompts based on actions_success_rate for full recursion enablement."
    target_files: ["a3x/planner.py"]
"""


class Planner:
    def __init__(self, *, thresholds: PlannerThresholds | None = None) -> None:
        self.thresholds = thresholds or PlannerThresholds()
        self.llm = OpenRouterLLMClient(model="x-ai/grok-4-fast:free")
        self.prompts = PromptTemplate()

    def propose(
        self,
        history: Dict[str, List[float]],
        *,
        patch_config_path: str,
        manual_config_path: str,
        tests_config_path: str,
        lint_config_path: str,
        capability_metrics: Optional[Dict[str, Dict[str, float | int | None]]] = None,
        last_errors: Optional[List[str]] = None,
    ) -> List[Seed]:
        from .seeds import Seed
        seeds: List[Seed] = []
        capability_metrics = capability_metrics or {}

        def latest(metric: str) -> Optional[float]:
            values = history.get(metric, [])
            return values[-1] if values else None

        def ensure_seed(id: str, seed: Seed) -> None:
            # Caller decides dedup with backlog; here we just list proposals
            seeds.append(seed)

        def ensure_error_seed(errors: List[str]) -> None:
            error_description = "; ".join(errors)
            seed_id = f"auto.error.reflection.{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            ensure_seed(
                seed_id,
                Seed(
                    id=seed_id,
                    goal=f"Investigar e corrigir erros de execução: {error_description}",
                    priority="high",
                    status="pending",
                    type="analysis",
                    config=manual_config_path,
                    max_steps=8,
                    metadata={
                        "description": "Seed gerada automaticamente para refletir sobre erros de execução.",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "errors": error_description,
                    },
                ),
            )

        if last_errors:
            ensure_error_seed(last_errors)

        apc = history.get("apply_patch_count", [])
        if apc and max(apc) == 0:
            ensure_seed(
                "auto.benchmark.diff",
                Seed(
                    id="auto.benchmark.diff",
                    goal="Atualize docs/seed_manifesto.md com uma seção curta sobre seeds automáticas.",
                    priority="high",
                    status="pending",
                    type="benchmark_diff",
                    config=patch_config_path,
                    max_steps=6,
                    metadata={
                        "description": "Seed auto-gerada para provocar um diff real",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                ),
            )

        # Always keep a report refresh around
        ensure_seed(
            "auto.benchmark.report",
            Seed(
                id="auto.benchmark.report",
                goal="Regenerar relatório SeedAI para testar ciclo seed-run.",
                priority="medium",
                status="pending",
                type="benchmark_report",
                config=manual_config_path,
                max_steps=3,
                metadata={
                    "description": "Seed auto-gerada para refresh de relatórios",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            ),
        )

        ap_success_last = latest("apply_patch_success_rate")
        diff_metrics = capability_metrics.get("core.diffing", {})
        diff_success_rate = diff_metrics.get("success_rate")
        if apc and max(apc) > 0:
            needs_patch_seed = False
            if (
                ap_success_last is None
                or ap_success_last < self.thresholds.apply_patch_success_rate
            ):
                needs_patch_seed = True
            if (
                isinstance(diff_success_rate, (int, float))
                and diff_success_rate < self.thresholds.apply_patch_success_rate
            ):
                needs_patch_seed = True
            if needs_patch_seed:
                ensure_seed(
                    "auto.patch.success",
                    Seed(
                        id="auto.patch.success",
                        goal="Atualizar configs/doc.md adicionando uma linha final de validação",
                        priority="high",
                        status="pending",
                        type="benchmark_diff",
                        config=patch_config_path,
                        max_steps=5,
                        metadata={
                            "description": "Seed corretiva para garantir sucesso no apply_patch",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    ),
                )

        actions_rate = history.get("actions_success_rate", [])
        if actions_rate and actions_rate[-1] < self.thresholds.actions_success_rate:
            ensure_seed(
                "auto.actions.recovery",
                Seed(
                    id="auto.actions.recovery",
                    goal="Verificar logs de seeds falhas e propor correção",
                    priority="medium",
                    status="pending",
                    type="analysis",
                    config=manual_config_path,
                    max_steps=4,
                    metadata={
                        "description": "Seed diagnóstica quando actions_success_rate cai.",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                ),
            )

        tests_rate = history.get("tests_success_rate", [])
        tests_count = history.get("tests_run_count", [])
        testing_metrics = capability_metrics.get("core.testing", {})
        tests_failures = testing_metrics.get("failures_detected")
        needs_tests_seed = False
        if tests_count and max(tests_count) > 0:
            last_tests_rate = tests_rate[-1] if tests_rate else None
            if (
                last_tests_rate is None
                or last_tests_rate < self.thresholds.tests_success_rate
            ):
                needs_tests_seed = True
            if isinstance(tests_failures, (int, float)) and tests_failures > 0:
                needs_tests_seed = True
        if needs_tests_seed:
            ensure_seed(
                "auto.tests.run",
                Seed(
                    id="auto.tests.run",
                    goal="Executar suíte de testes Pytest para garantir integridade.",
                    priority="high",
                    status="pending",
                    type="benchmark_tests",
                    config=tests_config_path,
                    max_steps=6,
                    metadata={
                        "description": "Seed auto-gerada para rodar pytest e diagnosticar falhas.",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                ),
            )

        lint_rate = history.get("lint_success_rate", [])
        lint_count = history.get("lint_run_count", [])
        if (
            lint_count
            and max(lint_count) > 0
            and (not lint_rate or lint_rate[-1] < self.thresholds.lint_success_rate)
        ):
            ensure_seed(
                "auto.lint.run",
                Seed(
                    id="auto.lint.run",
                    goal="Executar lint (ruff/black) para garantir estilo consistente.",
                    priority="medium",
                    status="pending",
                    type="benchmark_lint",
                    config=lint_config_path,
                    max_steps=6,
                    metadata={
                        "description": "Seed auto-gerada para rodar lint e corrigir estilo.",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                ),
            )

        # Integrate dynamic metric feedback and enhanced prompts for low actions_rate
        actions_rate = latest("actions_success_rate")
        if actions_rate is not None and actions_rate < 0.9:
            feedback = self.prompts.low_rate.format(rate=actions_rate)
            
            # Hook stateful retrieval for context-aware planning
            mock_state = AgentState(
                goal="Dynamic seed planning based on low actions_success_rate",
                history_snapshot=json.dumps({
                    "metrics_history": {k: v[-5:] for k, v in history.items() if len(v) >= 5},  # Recent history
                    "capability_metrics": capability_metrics or {},
                    "last_errors": last_errors or []
                }),
                iteration=1,
                max_iterations=1,
                seed_context="Planning seeds to improve actions success from recent sessions."
            )
            retriever = StatefulRetriever()
            session_context = retriever.retrieve_session_context(mock_state)
            
            # Summarize context, highlight recent failures or derivations
            context_parts = []
            derivation_count = 0
            for insight in session_context:
                context_parts.append(f"Insight: {insight.title} - {insight.content[:200]}... (similarity: {insight.similarity:.2f})")
                if insight.metadata.get("derivation_flagged"):
                    derivation_count += 1
            context_str = "\n".join(context_parts) if context_parts else "No relevant session context found."
            
            derivation_note = f" (Note: {derivation_count} insights show derivation changes - review for drift)" if derivation_count > 0 else ""
            
            prompt = f"""{self.prompts.base}

{feedback}

Recent session context{derivation_note}:
{context_str}

History summary: {json.dumps({k: v[-1] if isinstance(v, list) and v else None for k, v in history.items()}, indent=2) if history else 'No history available'}

Capability metrics: {json.dumps(capability_metrics or {}, indent=2)}

Last run errors: {', '.join(last_errors) if last_errors else 'None'}

Use chain-of-thought reasoning step-by-step:
1. Analyze gaps: Why is actions_success_rate low? Incorporate session context on recent failures (e.g., suboptimal action selection, lack of examples in prompts, insufficient planning for recursion). Consider recent failures in executor.py like patch applications or command executions.
2. Reason about improvements: For self-modify, detail how to add CoT and examples to prompts in planner.py to boost ROI and enable full real recursion (depth >=5). Also, suggest fixes for executor.py to improve patch success via better AST handling or simplified auto-commit logic. Use context for rationale.
3. Propose high-ROI seeds: Focus on benchmarks for practice (e.g., simple file writes/patches), refactors for prompt tuning in planner.py, and targeted fixes in executor.py for reliable commits and error recovery. Ensure seeds include safeguards and validation steps. Base on session context for continuity.

{self.prompts.examples}

Additional examples for executor fixes:
- id: meta.executor_patch_fix
  goal: "Simplificar lógica de auto-commit em a3x/executor.py removendo duplicações e garantindo git add/commit para low-risk self-modify sem prompts manuais."
  priority: high
  type: refactor
  config: configs/seed_patch.yaml
  max_steps: 8
  metadata:
    description: "Refactor to boost apply_patch_success_rate by streamlining auto-commit and AST fallback."
    target_files: ["a3x/executor.py"]

Available configs: patch={patch_config_path}, manual={manual_config_path}, tests={tests_config_path}, lint={lint_config_path}

Output ONLY a valid YAML list of 1-3 new seeds. Each seed must include: id (unique), goal (in Portuguese), priority ('high'), type ('refactor' or 'benchmark_diff' etc.), config (one of the available), max_steps (5-8), metadata (dict with 'description' and 'created_at' optional)."""

            try:
                response = self.llm.chat(prompt)
                parsed_seeds = yaml.safe_load(response)
                if isinstance(parsed_seeds, list):
                    for data in parsed_seeds:
                        if isinstance(data, dict) and 'id' in data and 'goal' in data:
                            data.setdefault('priority', 'high')
                            data.setdefault('status', 'pending')
                            data.setdefault('max_steps', 6)
                            data.setdefault('config', manual_config_path)
                            data.setdefault('metadata', {})
                            data['metadata'].setdefault('description', 'LLM-optimized seed for actions_rate improvement with session context')
                            data['metadata'].setdefault('created_at', datetime.now(timezone.utc).isoformat())
                            
                            # Ensure metadata values are strings
                            processed_metadata = {k: ", ".join(v) if isinstance(v, list) else str(v) for k, v in data['metadata'].items()}
                            data['metadata'] = processed_metadata

                            seeds.append(Seed(**data))
            except Exception:
                # Fallback to rule-based proposals if LLM fails
                pass

        return seeds



__all__ = ["Planner", "PlannerThresholds"]
