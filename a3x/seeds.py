"""Gerenciamento de seeds autônomas para o A3X."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml
from .planner import PlannerThresholds


_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_VALID_STATUS = {"pending", "in_progress", "completed", "failed"}


@dataclass
class Seed:
    id: str
    goal: str
    priority: str = "medium"
    status: str = "pending"
    type: str = "generic"
    config: Optional[str] = None
    max_steps: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3
    next_run_at: Optional[str] = None  # ISO-8601 em UTC
    last_error: Optional[str] = None
    last_iterations: Optional[int] = None
    last_success: Optional[bool] = None
    last_memories_reused: Optional[int] = None
    _fitness_cache: Optional[float] = field(default=None, repr=False, compare=False)

    def mark_status(self, status: str, *, notes: Optional[str] = None) -> None:
        if status not in _VALID_STATUS:
            raise ValueError(f"Status inválido: {status}")
        self.status = status
        entry = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if notes:
            entry["notes"] = notes
        self.history.append(entry)

    @property
    def priority_rank(self) -> int:
        return _PRIORITY_ORDER.get(self.priority, 99)

    @property
    def priority_numeric(self) -> float:
        mapping = {"high": 3.0, "medium": 2.0, "low": 1.0}
        return mapping.get(self.priority, 0.0)

    def compute_expected_gain(self) -> float:
        """Calcula o ganho esperado com base no histórico recente da seed e deltas de fitness."""
        
        # Original components
        if self.last_success is True:
            success_component = 1.0
        elif self.last_success is False:
            success_component = -0.5
        else:
            success_component = 0.5

        iterations_component = 1.0 / (1.0 + float(self.last_iterations or 0))
        memory_component = 0.2 * float(self.last_memories_reused or 0)
        
        # Fitness delta component - use fitness history to calculate expected gain
        fitness_delta_component = 0.0
        if 'fitness_delta' in self.metadata:
            try:
                fitness_delta = float(self.metadata.get('fitness_delta', 0))
                fitness_delta_component = fitness_delta  # Positive delta is good
            except (ValueError, TypeError):
                fitness_delta_component = 0.0
        
        # Apply a discount factor based on number of attempts (learning from repeated failures)
        attempts_discount = 1.0 / (1.0 + self.attempts * 0.2)  # 20% penalty per attempt
        
        return (success_component + iterations_component + memory_component + fitness_delta_component) * attempts_discount

    def compute_fitness(self) -> float:
        attempts_factor = max(self.attempts, 0)
        gain = self.compute_expected_gain()
        fitness = self.priority_numeric + gain / (1 + attempts_factor)
        self._fitness_cache = fitness
        return fitness

    @property
    def fitness(self) -> float:
        if self._fitness_cache is None:
            return self.compute_fitness()
        return self._fitness_cache


class SeedBacklog:
    def __init__(self, seeds: Iterable[Seed], path: Path) -> None:
        self._seeds = {seed.id: seed for seed in seeds}
        self.path = path

    @classmethod
    def load(cls, path: str | Path) -> "SeedBacklog":
        path_obj = Path(path)
        seeds: List[Seed] = []
        if path_obj.exists():
            with path_obj.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or []
            if not isinstance(raw, list):
                raise ValueError("Backlog de seeds deve ser uma lista")
            for entry in raw:
                seeds.append(_deserialize_seed(entry))
        return cls(seeds, path_obj)

    def save(self) -> None:
        data = []
        for seed in self._seeds.values():
            seed.compute_fitness()
            entry = {
                "id": seed.id,
                "goal": seed.goal,
                "priority": seed.priority,
                "status": seed.status,
                "type": seed.type,
                "config": seed.config,
                "max_steps": seed.max_steps,
                "metadata": seed.metadata,
                "history": seed.history,
                "attempts": seed.attempts,
                "max_attempts": seed.max_attempts,
                "next_run_at": seed.next_run_at,
                "last_error": seed.last_error,
                "last_iterations": seed.last_iterations,
                "last_success": seed.last_success,
                "last_memories_reused": seed.last_memories_reused,
                "fitness": seed.fitness,
            }
            data.append(entry)
        with self.path.open("w", encoding="utf-8") as fh:
            print(f"Debug: SeedBacklog.save - data before dump: {data}")
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)

    def add_seed(self, seed: Seed) -> None:
        """Adiciona uma seed ao backlog (sobrescreve se já existir mesmo id)."""
        self._seeds[seed.id] = seed
        self.save()

    def exists(self, seed_id: str) -> bool:
        return seed_id in self._seeds

    def list_pending(self) -> List[Seed]:
        return [seed for seed in self._seeds.values() if seed.status == "pending"]

    def list_all_ids(self) -> List[str]:
        return list(self._seeds.keys())

    def next_seed(self) -> Optional[Seed]:
        pending = self.list_pending()
        if not pending:
            return None
        # Filtra por janela de execução (respeita next_run_at quando definido)
        now = datetime.now(timezone.utc)
        eligible: List[Seed] = []
        for seed in pending:
            if seed.next_run_at:
                try:
                    ts = _parse_iso(seed.next_run_at)
                    if ts and ts > now:
                        continue
                except Exception:
                    pass
            eligible.append(seed)
        if not eligible:
            return None
        for seed in eligible:
            seed.compute_fitness()
        eligible.sort(
            key=lambda seed: (
                -seed.fitness,
                seed.priority_rank,
                str(seed.metadata.get("created_at", "")),
                seed.id,
            )
        )
        return eligible[0]

    def update_seed(self, seed: Seed) -> None:
        self._seeds[seed.id] = seed
        self.save()

    def mark_in_progress(self, seed_id: str) -> None:
        seed = self._seeds[seed_id]
        seed.mark_status("in_progress")
        seed.compute_fitness()
        self.save()

    def mark_completed(
        self,
        seed_id: str,
        *,
        notes: Optional[str] = None,
        iterations: Optional[int] = None,
        memories_reused: Optional[int] = None,
    ) -> None:
        seed = self._seeds[seed_id]
        seed.last_iterations = iterations
        seed.last_success = True
        seed.last_memories_reused = memories_reused
        seed.next_run_at = None
        seed.last_error = None
        seed.mark_status("completed", notes=notes)
        seed.compute_fitness()
        self.save()

    def mark_failed(
        self,
        seed_id: str,
        *,
        notes: Optional[str] = None,
        iterations: Optional[int] = None,
        memories_reused: Optional[int] = None,
    ) -> None:
        seed = self._seeds[seed_id]
        seed.last_error = notes
        seed.attempts += 1
        seed.last_iterations = iterations
        seed.last_success = False
        seed.last_memories_reused = memories_reused
        seed.mark_status("failed", notes=notes)
        # Reenfileira automaticamente com backoff se houver tentativas restantes
        if seed.attempts < seed.max_attempts:
            backoff_seconds = 60 * (2 ** (seed.attempts - 1))  # 60s, 120s, 240s...
            next_time = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
            seed.next_run_at = next_time.isoformat()
            seed.status = "pending"
            seed.mark_status(
                "pending", notes=f"requeue after backoff {backoff_seconds}s"
            )
        seed.compute_fitness()
        self.save()


def _deserialize_seed(entry: dict) -> Seed:
    required = {"id", "goal"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"Seed com campos ausentes: {missing}")
    history = entry.get("history") or []
    if not isinstance(history, list):
        raise ValueError("Campo history deve ser lista")
    return Seed(
        id=str(entry["id"]),
        goal=str(entry["goal"]),
        priority=str(entry.get("priority", "medium")),
        status=str(entry.get("status", "pending")),
        type=str(entry.get("type", "generic")),
        config=entry.get("config"),
        max_steps=entry.get("max_steps"),
        metadata={str(k): str(v) for k, v in (entry.get("metadata") or {}).items()},
        history=[{str(k): str(v) for k, v in item.items()} for item in history],
        attempts=int(entry.get("attempts", 0)),
        max_attempts=int(entry.get("max_attempts", 3)),
        next_run_at=entry.get("next_run_at"),
        last_error=entry.get("last_error"),
        last_iterations=(
            int(entry.get("last_iterations"))
            if entry.get("last_iterations") is not None
            else None
        ),
        last_success=entry.get("last_success"),
        last_memories_reused=(
            int(entry.get("last_memories_reused"))
            if entry.get("last_memories_reused") is not None
            else None
        ),
        _fitness_cache=(
            float(entry.get("fitness"))
            if entry.get("fitness") is not None
            else None
        ),
    )


def _parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except Exception:
        return None


class AutoSeeder:
    def __init__(self, thresholds: Optional[PlannerThresholds] = None) -> None:
        self.thresholds = thresholds or PlannerThresholds()

    def monitor_and_seed(self, metrics: Dict[str, float]) -> List[Seed]:
        seeds: List[Seed] = []
        gap_templates = {
            "actions_success_rate": {
                "threshold": self.thresholds.actions_success_rate,
                "goal": "Refatorar executor para maior sucesso em ações",
                "priority": "high",
                "config": "configs/sample.yaml",
                "type": "refactor",
                "description": "Refatorar executor para maior sucesso em ações"
            },
            "tests_success_rate": {
                "threshold": self.thresholds.tests_success_rate,
                "goal": "Expandir testgen para maior cobertura",
                "priority": "high",
                "config": "configs/sample.yaml",
                "type": "refactor",
                "description": "Expandir testgen para maior cobertura de testes"
            },
            "apply_patch_success_rate": {
                "threshold": self.thresholds.apply_patch_success_rate,
                "goal": "Adicionar verificações de segurança ao patch.py",
                "priority": "high",
                "config": "configs/sample.yaml",
                "type": "refactor",
                "description": "Adicionar safety checks ao patch.py"
            }
        }
        now_str = datetime.now(timezone.utc).isoformat()
        for key, template in gap_templates.items():
            value = metrics.get(key)
            if isinstance(value, (int, float)) and value < template["threshold"]:
                seed_id = f"auto.{key.replace('_', '.')}.{int(datetime.now(timezone.utc).timestamp())}"
                seed = Seed(
                    id=seed_id,
                    goal=template["goal"],
                    priority=template["priority"],
                    type=template["type"],
                    config=template["config"],
                    max_steps=8,
                    metadata={
                        "description": template["description"],
                        "created_at": now_str,
                        "triggered_by": key
                    }
                )
                seeds.append(seed)
        return seeds


__all__ = ["Seed", "SeedBacklog", "AutoSeeder"]
