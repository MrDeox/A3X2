"""Meta capabilities manager: generate seeds when advanced states are reached."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from .capabilities import CapabilityRegistry
from .seeds import Seed


class MetaCapabilityPlanner:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    def propose(
        self,
        capability_metrics: Dict[str, Dict[str, float | int | None]],
        *,
        backlog_existing: Iterable[str],
        config_map: Dict[str, str],
    ) -> List[Seed]:
        seeds: List[Seed] = []
        existing = set(backlog_existing)

        for capability in self.registry.list():
            if capability.category != "meta":
                continue

            if not self._requirements_met(capability.requirements, capability_metrics):
                continue

            activation = capability.activation or {}
            seed_id = activation.get("seed_id") or f"meta.{capability.id}"
            if seed_id in existing:
                continue

            goal = activation.get("goal") or capability.description
            seed_type = activation.get("type", "meta")
            priority = activation.get("priority", "medium")
            config_key = activation.get("config", "manual")
            config_path = config_map.get(config_key, config_map.get("manual"))
            metadata = {
                "meta_capability": capability.id,
                "description": capability.description,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            max_steps_value: Optional[int] = None
            if activation.get("max_steps") is not None:
                try:
                    max_steps_value = int(activation["max_steps"])
                except (TypeError, ValueError):
                    max_steps_value = None

            seed = Seed(
                id=seed_id,
                goal=goal,
                priority=priority,
                status="pending",
                type=seed_type,
                config=config_path,
                max_steps=max_steps_value,
                metadata=metadata,
            )
            seeds.append(seed)

        return seeds

    def _requirements_met(
        self,
        requirements: Dict[str, str],
        capability_metrics: Dict[str, Dict[str, float | int | None]],
    ) -> bool:
        if not requirements:
            return False
        for capability_id, maturity_required in requirements.items():
            try:
                capability = self.registry.get(capability_id)
            except KeyError:
                return False
            if (
                capability.maturity not in {maturity_required, "advanced"}
                and capability.maturity != maturity_required
            ):
                return False
            metrics = capability_metrics.get(capability_id, {})
            if capability.maturity == maturity_required:
                continue
            # fallback: ensure metrics recorded if maturity not yet updated
            if not metrics:
                return False
        return True


__all__ = ["MetaCapabilityPlanner"]
