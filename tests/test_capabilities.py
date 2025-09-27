from pathlib import Path

import pytest

from a3x.capabilities import CapabilityRegistry


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

