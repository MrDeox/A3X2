"""SeedAI capability registry utilities."""

from __future__ import annotations

import builtins
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Capability:
    """Represents a capability tracked by the SeedAI capability graph."""

    id: str
    name: str
    category: str  # "horizontal" or "vertical"
    description: str
    maturity: str
    metrics: dict[str, float | None] = field(default_factory=dict)
    seeds: list[str] = field(default_factory=list)
    requirements: dict[str, str] = field(default_factory=dict)
    activation: dict[str, str] = field(default_factory=dict)


class CapabilityRegistry:
    """Loads and serializes the capability graph from YAML files."""

    def __init__(
        self, capabilities: Iterable[Capability], raw_entries: dict[str, dict]
    ) -> None:
        self._by_id = {cap.id: cap for cap in capabilities}
        self._raw_entries = raw_entries

    @classmethod
    def from_yaml(cls, path: str | Path) -> CapabilityRegistry:
        entries = _read_yaml(Path(path))
        capabilities = [_deserialize_capability(item, path) for item in entries]
        raw_map = {
            str(entry["id"]): entry
            for entry in entries
            if isinstance(entry, dict) and "id" in entry
        }
        return cls(capabilities, raw_map)

    def list(self) -> builtins.list[Capability]:
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

    def update_metrics(self, updates: dict[str, dict[str, float | None]]) -> None:
        for cap_id, metric_values in updates.items():
            if cap_id not in self._by_id:
                continue
            capability = self._by_id[cap_id]
            for key, value in metric_values.items():
                capability.metrics[key] = value
        raw = self._raw_entries.get(cap_id)
        if raw is not None:
            raw_metrics = raw.setdefault("metrics", {})
            for key, value in metric_values.items():
                raw_metrics[key] = value

    def update_maturity(self, updates: dict[str, str]) -> None:
        for cap_id, maturity in updates.items():
            if cap_id not in self._by_id:
                continue
            capability = self._by_id[cap_id]
            capability.maturity = maturity
            raw = self._raw_entries.get(cap_id)
            if raw is not None:
                raw["maturity"] = maturity

    def to_yaml(self, path: str | Path, header_comment: str | None = None) -> None:
        data = [self._raw_entries[cap.id] for cap in self.list()]
        output = ""
        if header_comment:
            output += header_comment.rstrip() + "\n"
        output += yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        Path(path).write_text(output, encoding="utf-8")


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

    requirements = entry.get("requirements") or {}
    if not isinstance(requirements, dict):
        raise ValueError(f"Campo requirements deve ser objeto em {entry['id']}")

    activation = entry.get("activation") or {}
    if not isinstance(activation, dict):
        raise ValueError(f"Campo activation deve ser objeto em {entry['id']}")

    metrics_normalized: dict[str, float | None] = {}
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
        requirements={str(k): str(v) for k, v in requirements.items()},
        activation={str(k): str(v) for k, v in activation.items()},
    )


__all__ = [
    "Capability",
    "CapabilityRegistry",
]
