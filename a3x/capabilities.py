"""SeedAI capability registry utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


@dataclass
class Capability:
    """Represents a capability tracked by the SeedAI capability graph."""

    id: str
    name: str
    category: str  # "horizontal" or "vertical"
    description: str
    maturity: str
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    seeds: List[str] = field(default_factory=list)


class CapabilityRegistry:
    """Loads and serializes the capability graph from YAML files."""

    def __init__(self, capabilities: Iterable[Capability]) -> None:
        self._by_id = {cap.id: cap for cap in capabilities}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CapabilityRegistry":
        data = _read_yaml(Path(path))
        capabilities = [_deserialize_capability(item, path) for item in data]
        return cls(capabilities)

    def list(self) -> List[Capability]:
        return list(self._by_id.values())

    def get(self, capability_id: str) -> Capability:
        try:
            return self._by_id[capability_id]
        except KeyError as exc:  # pragma: no cover - simples acesso
            raise KeyError(f"Capability não encontrada: {capability_id}") from exc

    def summary(self) -> str:
        lines = []
        for cap in self._by_id.values():
            lines.append(f"- {cap.id} ({cap.category}) :: {cap.name} [{cap.maturity}]")
            lines.append(f"  {cap.description}")
            if cap.seeds:
                lines.append("  Seeds: " + "; ".join(cap.seeds))
        return "\n".join(lines)


def _read_yaml(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de capacidades não encontrado: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []
    if not isinstance(data, list):
        raise ValueError("Capacidades devem ser uma lista de objetos")
    return data


def _deserialize_capability(entry: dict, source: Path) -> Capability:
    required_fields = ["id", "name", "category", "description", "maturity"]
    missing = [field for field in required_fields if field not in entry]
    if missing:
        raise ValueError(f"Entrada inválida em {source}: campos ausentes {missing}")

    metrics = entry.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise ValueError(f"Campo metrics deve ser objeto em {entry['id']}")

    seeds = entry.get("seeds") or []
    if not isinstance(seeds, list):
        raise ValueError(f"Campo seeds deve ser lista em {entry['id']}")

    metrics_normalized: Dict[str, Optional[float]] = {}
    for key, value in metrics.items():
        if value is None:
            metrics_normalized[str(key)] = None
        elif isinstance(value, (int, float)):
            metrics_normalized[str(key)] = float(value)
        else:
            raise ValueError(
                f"Valor inválido em metrics[{key}] para capability {entry['id']}: {value}"
            )

    return Capability(
        id=str(entry["id"]),
        name=str(entry["name"]),
        category=str(entry["category"]),
        description=str(entry["description"]),
        maturity=str(entry["maturity"]),
        metrics=metrics_normalized,
        seeds=[str(seed) for seed in seeds],
    )


__all__ = [
    "Capability",
    "CapabilityRegistry",
]
