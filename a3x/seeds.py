"""Gerenciamento de seeds autônomas para o A3X."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


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
            }
            data.append(entry)
        with self.path.open("w", encoding="utf-8") as fh:
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
        eligible.sort(
            key=lambda seed: (seed.priority_rank, seed.metadata.get("created_at", ""))
        )
        return eligible[0]

    def update_seed(self, seed: Seed) -> None:
        self._seeds[seed.id] = seed
        self.save()

    def mark_in_progress(self, seed_id: str) -> None:
        seed = self._seeds[seed_id]
        seed.mark_status("in_progress")
        self.save()

    def mark_completed(self, seed_id: str, *, notes: Optional[str] = None) -> None:
        seed = self._seeds[seed_id]
        seed.mark_status("completed", notes=notes)
        self.save()

    def mark_failed(self, seed_id: str, *, notes: Optional[str] = None) -> None:
        seed = self._seeds[seed_id]
        seed.last_error = notes
        seed.attempts += 1
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


__all__ = ["Seed", "SeedBacklog"]
