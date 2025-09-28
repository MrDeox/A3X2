from pathlib import Path

import pytest

from a3x.capabilities import CapabilityRegistry
from a3x.capability_metrics import compute_capability_metrics


def test_capability_registry_load(tmp_path: Path) -> None:
    capability_yaml = tmp_path / "caps.yaml"
    capability_yaml.write_text(
        """
- id: sample
  name: Sample Capability
  category: vertical
  description: Something useful
  maturity: baseline
  metrics:
    success_rate: 0.75
  seeds:
    - Improve coverage
  requirements:
    core.testing: established
  activation:
    goal: Advanced step
    priority: high
    type: meta
        """,
        encoding="utf-8",
    )

    registry = CapabilityRegistry.from_yaml(capability_yaml)
    capabilities = registry.list()

    assert len(capabilities) == 1
    capability = capabilities[0]
    assert capability.id == "sample"
    assert capability.metrics["success_rate"] == 0.75
    assert "Improve coverage" in capability.seeds
    assert capability.requirements["core.testing"] == "established"
    assert capability.activation["goal"] == "Advanced step"


def test_capability_registry_invalid_metrics(tmp_path: Path) -> None:
    capability_yaml = tmp_path / "caps.yaml"
    capability_yaml.write_text(
        """
- id: invalid
  name: Invalid Capability
  category: horizontal
  description: invalid metrics
  maturity: baseline
  metrics:
    success_rate: "high"
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        CapabilityRegistry.from_yaml(capability_yaml)


def test_compute_capability_metrics(tmp_path: Path) -> None:
    log = tmp_path / "run_evaluations.jsonl"
    log.write_text(
        """
{"capabilities": ["core.diffing", "core.testing"], "completed": true, "iterations": 4, "metrics": {"apply_patch_success_rate": 1.0, "tests_run_count": 1.0, "tests_success_rate": 0.5}}
{"capabilities": ["core.diffing", "horiz.python"], "completed": false, "iterations": 6, "metrics": {"apply_patch_success_rate": 0.0}}
{"capabilities": ["horiz.docs"], "completed": true, "iterations": 2, "metrics": {}}
        """.strip(),
        encoding="utf-8",
    )

    metrics = compute_capability_metrics(log)

    assert metrics["core.diffing"]["success_rate"] == 0.5
    assert metrics["core.testing"]["auto_trigger_rate"] == 1.0
    assert metrics["core.testing"]["failures_detected"] == 1
    assert metrics["horiz.python"]["tasks_completed"] == 0
    assert metrics["horiz.docs"]["docs_generated"] == 1
    assert metrics["core.diffing"]["runs"] == 2
    assert metrics["core.testing"]["completed_runs"] == 1
