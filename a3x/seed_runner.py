"""Executa seeds autônomas utilizando o agente A3X."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .agent import AgentOrchestrator
from .config import load_config
from .llm import build_llm_client
from .seeds import SeedBacklog


@dataclass
class SeedRunResult:
    seed_id: str
    status: str
    completed: bool
    notes: str = ""
    iterations: int = 0
    memories_reused: int = 0


class SeedRunner:
    def __init__(self, backlog_path: str | Path) -> None:
        self.backlog = SeedBacklog.load(backlog_path)

    def run_next(
        self, *, default_config: str | Path, max_steps_override: Optional[int] = None
    ) -> Optional[SeedRunResult]:
        seed = self.backlog.next_seed()
        if not seed:
            return None

        self.backlog.mark_in_progress(seed.id)
        # Prefer sample.yaml for meta/recursivity to enable LLM-driven runs
        if seed.type == "meta":
            config_path = Path("configs/sample.yaml")
        else:
            config_path = Path(seed.config or default_config)
        config = load_config(config_path)
        if seed.max_steps and not max_steps_override:
            config.limits.max_iterations = seed.max_steps
        elif max_steps_override:
            config.limits.max_iterations = max_steps_override

        llm_client = build_llm_client(config.llm)
        orchestrator = AgentOrchestrator(config, llm_client)
        result = orchestrator.run(seed.goal)

        notes = ""
        if result.errors:
            notes = "; ".join(result.errors)
        if result.completed:
            self.backlog.mark_completed(
                seed.id,
                notes=notes,
                iterations=result.iterations,
                memories_reused=result.memories_reused,
            )
            return SeedRunResult(
                seed_id=seed.id,
                status="completed",
                completed=True,
                notes=notes,
                iterations=result.iterations,
                memories_reused=result.memories_reused,
            )

        self.backlog.mark_failed(
            seed.id,
            notes=notes or "Seed não concluída",
            iterations=result.iterations,
            memories_reused=result.memories_reused,
        )
        return SeedRunResult(
            seed_id=seed.id,
            status="failed",
            completed=False,
            notes=notes,
            iterations=result.iterations,
            memories_reused=result.memories_reused,
        )


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Executa a próxima seed autônoma do backlog"
    )
    parser.add_argument(
        "--backlog", default="seed/backlog.yaml", help="Arquivo YAML do backlog"
    )
    parser.add_argument(
        "--config",
        default="configs/sample.yaml",
        help="Config padrão caso seed não defina a sua",
    )
    parser.add_argument("--max-steps", type=int, help="Override de max iterations")
    args = parser.parse_args(argv)

    runner = SeedRunner(args.backlog)
    result = runner.run_next(
        default_config=args.config, max_steps_override=args.max_steps
    )
    if result is None:
        print("Nenhuma seed pendente")
        return 0
    print(
        f"Seed {result.seed_id} -> {result.status} ({'completa' if result.completed else 'falhou'})"
    )
    if result.notes:
        print(f"Notas: {result.notes}")
    return 0 if result.completed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
