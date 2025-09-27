"""Simple planner that proposes auto-seeds based on metric gaps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

from .seeds import Seed


@dataclass
class PlannerThresholds:
    apply_patch_success_rate: float = 0.8
    actions_success_rate: float = 0.8
    tests_success_rate: float = 0.9
    lint_success_rate: float = 0.9


class Planner:
    def __init__(self, *, thresholds: PlannerThresholds | None = None) -> None:
        self.thresholds = thresholds or PlannerThresholds()

    def propose(
        self,
        history: Dict[str, List[float]],
        *,
        patch_config_path: str,
        manual_config_path: str,
        tests_config_path: str,
        lint_config_path: str,
    ) -> List[Seed]:
        seeds: List[Seed] = []

        def ensure_seed(id: str, seed: Seed) -> None:
            # Caller decides dedup with backlog; here we just list proposals
            seeds.append(seed)

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

        ap_success = history.get("apply_patch_success_rate", [])
        if apc and max(apc) > 0 and (not ap_success or ap_success[-1] < self.thresholds.apply_patch_success_rate):
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
        if tests_count and max(tests_count) > 0 and (not tests_rate or tests_rate[-1] < self.thresholds.tests_success_rate):
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
        if lint_count and max(lint_count) > 0 and (not lint_rate or lint_rate[-1] < self.thresholds.lint_success_rate):
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

        return seeds


__all__ = ["Planner", "PlannerThresholds"]
