"""Executa seeds autônomas utilizando o agente A3X."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent import AgentOrchestrator
from .config import load_config
from .llm import build_llm_client
from .meta_capabilities import SkillProposal
from .seeds import Seed, SeedBacklog
from .skills.skill_creator import SkillCreator


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

        notes_parts: List[str] = []
        if result.errors:
            notes_parts.append("; ".join(result.errors))

        if result.completed:
            self.backlog.mark_completed(
                seed.id,
                notes=notes,
                iterations=result.iterations,
                memories_reused=result.memories_reused,
            )
            if seed.type == "skill_creation":
                success, message = self._handle_skill_creation(seed)
                if message:
                    notes_parts.append(message)
                if not success:
                    notes = "; ".join(part for part in notes_parts if part)
                    self.backlog.mark_failed(
                        seed.id,
                        notes=notes or "Falha ao materializar skill",
                    )
                    return SeedRunResult(
                        seed_id=seed.id,
                        status="failed",
                        completed=False,
                        notes=notes,
                    )

            notes = "; ".join(part for part in notes_parts if part)
            self.backlog.mark_completed(seed.id, notes=notes or None)
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
        notes = "; ".join(part for part in notes_parts if part)
        self.backlog.mark_failed(seed.id, notes=notes or "Seed não concluída")
        return SeedRunResult(
            seed_id=seed.id,
            status="failed",
            completed=False,
            notes=notes,
            iterations=result.iterations,
            memories_reused=result.memories_reused,
        )

    def _handle_skill_creation(self, seed: Seed) -> Tuple[bool, str]:
        try:
            proposal = self._load_skill_proposal(seed)
        except Exception as exc:
            return False, f"Falha ao carregar SkillProposal: {exc}"

        creator = SkillCreator(Path.cwd())
        try:
            return creator.create_skill_from_proposal(proposal)
        except Exception as exc:  # pragma: no cover - segurança adicional
            return False, f"Erro ao criar skill: {exc}"

    def _load_skill_proposal(self, seed: Seed) -> SkillProposal:
        metadata = seed.metadata or {}
        payload = self._extract_proposal_payload(metadata, seed)
        return self._build_skill_proposal(payload, metadata)

    def _extract_proposal_payload(
        self, metadata: Dict[str, str], seed: Seed
    ) -> Dict[str, Any]:
        for key in ("proposal_json", "skill_proposal_json", "skill_proposal"):
            raw = metadata.get(key)
            if raw:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON inválido em {key}: {exc}") from exc

        proposal_path = self._resolve_proposal_path(metadata, seed)
        if proposal_path:
            try:
                return json.loads(proposal_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Arquivo de proposta inválido em {proposal_path}: {exc}"
                ) from exc

        required_fields = [
            "id",
            "name",
            "description",
            "implementation_plan",
            "required_dependencies",
            "estimated_effort",
            "priority",
            "rationale",
            "target_domain",
            "created_at",
        ]
        if all(metadata.get(field) for field in required_fields):
            payload = {field: metadata[field] for field in required_fields}
            blueprint = metadata.get("blueprint_path") or metadata.get("blueprint_file")
            if blueprint:
                payload["blueprint_path"] = blueprint
            return payload

        raise ValueError("Metadados da seed não contêm proposta de skill")

    def _resolve_proposal_path(
        self, metadata: Dict[str, str], seed: Seed
    ) -> Optional[Path]:
        path_keys = (
            "proposal_record",
            "proposal_file",
            "proposal_path",
            "skill_proposal_path",
            "proposal_json_path",
        )
        for key in path_keys:
            value = metadata.get(key)
            if not value:
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = Path.cwd() / value
            if candidate.exists():
                return candidate

        proposal_id = (
            metadata.get("proposal_id")
            or metadata.get("proposal")
            or metadata.get("id")
            or seed.metadata.get("proposal_id")
        )
        if proposal_id:
            candidate = Path.cwd() / "seed" / "skills" / f"{proposal_id}.json"
            if candidate.exists():
                return candidate
        return None

    def _build_skill_proposal(
        self, payload: Dict[str, Any], metadata: Dict[str, str]
    ) -> SkillProposal:
        fields = [
            "id",
            "name",
            "description",
            "implementation_plan",
            "required_dependencies",
            "estimated_effort",
            "priority",
            "rationale",
            "target_domain",
            "created_at",
        ]
        normalized: Dict[str, Any] = {}
        for field in fields:
            if field in payload and payload[field] not in (None, ""):
                normalized[field] = payload[field]
            elif metadata.get(field):
                normalized[field] = metadata[field]

        missing = [field for field in fields if field not in normalized]
        if missing:
            raise ValueError(
                f"Campos ausentes para SkillProposal: {', '.join(sorted(missing))}"
            )

        normalized["required_dependencies"] = self._parse_dependencies(
            normalized["required_dependencies"]
        )
        normalized["estimated_effort"] = self._parse_float(
            normalized["estimated_effort"], "estimated_effort"
        )

        blueprint = (
            payload.get("blueprint_path")
            or payload.get("blueprint_file")
            or metadata.get("blueprint_path")
            or metadata.get("blueprint_file")
        )
        if blueprint:
            normalized["blueprint_path"] = blueprint

        return SkillProposal(**normalized)

    @staticmethod
    def _parse_dependencies(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
            return [item.strip() for item in stripped.split(",") if item.strip()]
        raise ValueError("Lista de dependências inválida para SkillProposal")

    @staticmethod
    def _parse_float(value: Any, field_name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Valor inválido para {field_name}: {value}") from exc


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
