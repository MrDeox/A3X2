"""Comprehensive tests for the capabilities module."""

import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from a3x.capabilities import Capability, CapabilityRegistry, _deserialize_capability, _read_yaml


class TestCapability:
    """Test cases for the Capability dataclass."""

    def test_capability_creation_minimal(self) -> None:
        """Test creating a capability with minimal required fields."""
        cap = Capability(
            id="test.capability",
            name="Test Capability",
            category="horizontal",
            description="A test capability",
            maturity="experimental"
        )

        assert cap.id == "test.capability"
        assert cap.name == "Test Capability"
        assert cap.category == "horizontal"
        assert cap.description == "A test capability"
        assert cap.maturity == "experimental"
        assert cap.metrics == {}
        assert cap.seeds == []
        assert cap.requirements == {}
        assert cap.activation == {}

    def test_capability_creation_full(self) -> None:
        """Test creating a capability with all fields."""
        cap = Capability(
            id="test.full",
            name="Full Test Capability",
            category="vertical",
            description="A full test capability",
            maturity="stable",
            metrics={"accuracy": 0.95, "latency": None, "throughput": 100.0},
            seeds=["seed1", "seed2"],
            requirements={"python": ">=3.8"},
            activation={"condition": "enabled"}
        )

        assert cap.id == "test.full"
        assert cap.name == "Full Test Capability"
        assert cap.category == "vertical"
        assert cap.maturity == "stable"
        assert cap.metrics == {"accuracy": 0.95, "latency": None, "throughput": 100.0}
        assert cap.seeds == ["seed1", "seed2"]
        assert cap.requirements == {"python": ">=3.8"}
        assert cap.activation == {"condition": "enabled"}

    def test_capability_field_types_validation(self) -> None:
        """Test that capability fields are properly typed."""
        cap = Capability(
            id="type.test",
            name="Type Test",
            category="horizontal",
            description="Testing types",
            maturity="beta"
        )

        # Test that fields maintain their types after creation
        assert isinstance(cap.metrics, dict)
        assert isinstance(cap.seeds, list)
        assert isinstance(cap.requirements, dict)
        assert isinstance(cap.activation, dict)


class TestCapabilityRegistry:
    """Test cases for the CapabilityRegistry class."""

    def test_registry_creation_empty(self) -> None:
        """Test creating an empty registry."""
        registry = CapabilityRegistry([], {})

        assert len(registry.list()) == 0
        assert registry._by_id == {}
        assert registry._raw_entries == {}

    def test_registry_creation_with_capabilities(self) -> None:
        """Test creating a registry with capabilities."""
        cap1 = Capability(
            id="test.cap1",
            name="Capability 1",
            category="horizontal",
            description="First capability",
            maturity="experimental"
        )
        cap2 = Capability(
            id="test.cap2",
            name="Capability 2",
            category="vertical",
            description="Second capability",
            maturity="beta"
        )

        raw_entries = {
            "test.cap1": {"id": "test.cap1", "name": "Capability 1"},
            "test.cap2": {"id": "test.cap2", "name": "Capability 2"}
        }

        registry = CapabilityRegistry([cap1, cap2], raw_entries)

        capabilities = registry.list()
        assert len(capabilities) == 2
        assert cap1 in capabilities
        assert cap2 in capabilities

    def test_registry_from_yaml_valid(self) -> None:
        """Test creating a registry from valid YAML content."""
        yaml_content = """
- id: "test.cap1"
  name: "Test Capability 1"
  category: "horizontal"
  description: "First test capability"
  maturity: "experimental"
  metrics:
    accuracy: 0.85
  seeds:
    - "seed1"
  requirements:
    python: ">=3.8"
  activation:
    enabled: true

- id: "test.cap2"
  name: "Test Capability 2"
  category: "vertical"
  description: "Second test capability"
  maturity: "beta"
  metrics:
    latency: null
    throughput: 100.5
  seeds: []
  requirements: {}
  activation: {}
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            registry = CapabilityRegistry.from_yaml(temp_path)

            capabilities = registry.list()
            assert len(capabilities) == 2

            cap1 = registry.get("test.cap1")
            assert cap1.name == "Test Capability 1"
            assert cap1.category == "horizontal"
            assert cap1.metrics["accuracy"] == 0.85
            assert cap1.seeds == ["seed1"]
            assert cap1.requirements == {"python": ">=3.8"}

            cap2 = registry.get("test.cap2")
            assert cap2.name == "Test Capability 2"
            assert cap2.category == "vertical"
            assert cap2.metrics["latency"] is None
            assert cap2.metrics["throughput"] == 100.5

        finally:
            Path(temp_path).unlink()

    def test_registry_from_yaml_missing_fields(self) -> None:
        """Test registry creation with missing required fields."""
        yaml_content = """
- id: "incomplete.cap"
  name: "Incomplete Capability"
  # Missing category, description, maturity
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="campos ausentes"):
                CapabilityRegistry.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_registry_from_yaml_invalid_metrics_type(self) -> None:
        """Test registry creation with invalid metrics field type."""
        yaml_content = """
- id: "invalid.metrics"
  name: "Invalid Metrics"
  category: "horizontal"
  description: "Test invalid metrics"
  maturity: "experimental"
  metrics: "not_a_dict"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Campo metrics deve ser objeto"):
                CapabilityRegistry.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_registry_from_yaml_invalid_seeds_type(self) -> None:
        """Test registry creation with invalid seeds field type."""
        yaml_content = """
- id: "invalid.seeds"
  name: "Invalid Seeds"
  category: "horizontal"
  description: "Test invalid seeds"
  maturity: "experimental"
  seeds: "not_a_list"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Campo seeds deve ser lista"):
                CapabilityRegistry.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_registry_get_existing_capability(self) -> None:
        """Test getting an existing capability."""
        cap = Capability(
            id="test.get",
            name="Get Test",
            category="horizontal",
            description="Test getting capability",
            maturity="experimental"
        )
        registry = CapabilityRegistry([cap], {"test.get": {"id": "test.get"}})

        retrieved = registry.get("test.get")
        assert retrieved == cap

    def test_registry_get_nonexistent_capability(self) -> None:
        """Test getting a nonexistent capability."""
        registry = CapabilityRegistry([], {})

        with pytest.raises(KeyError, match="Capability não encontrada"):
            registry.get("nonexistent.cap")

    def test_registry_summary(self) -> None:
        """Test generating registry summary."""
        cap1 = Capability(
            id="test.cap1",
            name="Capability 1",
            category="horizontal",
            description="First capability",
            maturity="experimental",
            seeds=["seed1", "seed2"]
        )
        cap2 = Capability(
            id="test.cap2",
            name="Capability 2",
            category="vertical",
            description="Second capability",
            maturity="stable"
        )

        registry = CapabilityRegistry([cap1, cap2], {
            "test.cap1": {"id": "test.cap1"},
            "test.cap2": {"id": "test.cap2"}
        })

        summary = registry.summary()
        lines = summary.split('\n')

        assert any("test.cap1 (horizontal) :: Capability 1 [experimental]" in line for line in lines)
        assert any("test.cap2 (vertical) :: Capability 2 [stable]" in line for line in lines)
        assert any("First capability" in line for line in lines)
        assert any("Second capability" in line for line in lines)
        assert any("Seeds: seed1; seed2" in line for line in lines)

    def test_registry_update_metrics(self) -> None:
        """Test updating capability metrics."""
        cap = Capability(
            id="test.metrics",
            name="Metrics Test",
            category="horizontal",
            description="Test metrics update",
            maturity="experimental",
            metrics={"accuracy": 0.8, "latency": 100.0}
        )

        raw_entries = {"test.metrics": {"id": "test.metrics", "metrics": {"accuracy": 0.8}}}
        registry = CapabilityRegistry([cap], raw_entries)

        # Update metrics
        updates = {"test.metrics": {"accuracy": 0.9, "throughput": 200.0}}
        registry.update_metrics(updates)

        # Check in-memory capability
        assert cap.metrics["accuracy"] == 0.9
        assert cap.metrics["throughput"] == 200.0
        assert cap.metrics["latency"] == 100.0  # Unchanged

        # Check raw entries
        assert raw_entries["test.metrics"]["metrics"]["accuracy"] == 0.9
        assert raw_entries["test.metrics"]["metrics"]["throughput"] == 200.0

    def test_registry_update_metrics_nonexistent_capability(self) -> None:
        """Test updating metrics for nonexistent capability."""
        registry = CapabilityRegistry([], {})

        # Should not raise error, just skip
        updates = {"nonexistent.cap": {"accuracy": 0.9}}
        registry.update_metrics(updates)  # Should complete without error

    def test_registry_update_maturity(self) -> None:
        """Test updating capability maturity."""
        cap = Capability(
            id="test.maturity",
            name="Maturity Test",
            category="horizontal",
            description="Test maturity update",
            maturity="experimental"
        )

        raw_entries = {"test.maturity": {"id": "test.maturity", "maturity": "experimental"}}
        registry = CapabilityRegistry([cap], raw_entries)

        # Update maturity
        updates = {"test.maturity": "stable"}
        registry.update_maturity(updates)

        # Check in-memory capability
        assert cap.maturity == "stable"

        # Check raw entries
        assert raw_entries["test.maturity"]["maturity"] == "stable"

    def test_registry_update_maturity_nonexistent_capability(self) -> None:
        """Test updating maturity for nonexistent capability."""
        registry = CapabilityRegistry([], {})

        # Should not raise error, just skip
        updates = {"nonexistent.cap": "stable"}
        registry.update_maturity(updates)  # Should complete without error

    def test_registry_to_yaml(self) -> None:
        """Test serializing registry to YAML."""
        cap = Capability(
            id="test.yaml",
            name="YAML Test",
            category="horizontal",
            description="Test YAML serialization",
            maturity="experimental",
            metrics={"accuracy": 0.95}
        )

        raw_entries = {
            "test.yaml": {
                "id": "test.yaml",
                "name": "YAML Test",
                "category": "horizontal",
                "description": "Test YAML serialization",
                "maturity": "experimental",
                "metrics": {"accuracy": 0.95}
            }
        }

        registry = CapabilityRegistry([cap], raw_entries)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            # Serialize to YAML
            registry.to_yaml(temp_path, header_comment="# Test Header")

            # Read back and verify
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assert content.startswith("# Test Header\n")
            data = yaml.safe_load(content[content.find('-'):])  # Skip header
            assert len(data) == 1
            assert data[0]["id"] == "test.yaml"
            assert data[0]["name"] == "YAML Test"
            assert data[0]["metrics"]["accuracy"] == 0.95

        finally:
            Path(temp_path).unlink()


class TestReadYaml:
    """Test cases for the _read_yaml helper function."""

    def test_read_yaml_success(self) -> None:
        """Test successful YAML reading."""
        yaml_content = """
- id: "test.cap"
  name: "Test Capability"
  category: "horizontal"
  description: "Test YAML reading"
  maturity: "experimental"
"""

        with patch("a3x.capabilities.Path.open", new_callable=mock_open, read_data=yaml_content):
            with patch("a3x.capabilities.Path.exists", return_value=True):
                result = _read_yaml(Path("test.yaml"))

                assert len(result) == 1
                assert result[0]["id"] == "test.cap"
                assert result[0]["name"] == "Test Capability"

    def test_read_yaml_file_not_found(self) -> None:
        """Test reading nonexistent YAML file."""
        with patch("a3x.capabilities.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Arquivo de capacidades não encontrado"):
                _read_yaml(Path("nonexistent.yaml"))

    def test_read_yaml_empty_file(self) -> None:
        """Test reading empty YAML file."""
        with patch("a3x.capabilities.Path.open", new_callable=mock_open, read_data=""):
            with patch("a3x.capabilities.Path.exists", return_value=True):
                result = _read_yaml(Path("empty.yaml"))
                assert result == []

    def test_read_yaml_invalid_yaml(self) -> None:
        """Test reading invalid YAML content."""
        invalid_yaml = """
- id: "invalid.cap"
  name: "Invalid YAML
  unclosed: string
"""

        with patch("a3x.capabilities.Path.open", new_callable=mock_open, read_data=invalid_yaml):
            with patch("a3x.capabilities.Path.exists", return_value=True):
                with patch("a3x.capabilities.yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
                    with pytest.raises(yaml.YAMLError):
                        _read_yaml(Path("invalid.yaml"))

    def test_read_yaml_non_list_root(self) -> None:
        """Test reading YAML with non-list root."""
        yaml_content = "id: not-a-list"

        with patch("a3x.capabilities.Path.open", new_callable=mock_open, read_data=yaml_content):
            with patch("a3x.capabilities.Path.exists", return_value=True):
                with patch("a3x.capabilities.yaml.safe_load", return_value={"id": "not-a-list"}):
                    with pytest.raises(ValueError, match="Capacidades devem ser uma lista"):
                        _read_yaml(Path("non-list.yaml"))


class TestDeserializeCapability:
    """Test cases for the _deserialize_capability helper function."""

    def test_deserialize_capability_minimal(self) -> None:
        """Test deserializing capability with minimal fields."""
        entry = {
            "id": "test.minimal",
            "name": "Minimal Test",
            "category": "horizontal",
            "description": "Minimal capability for testing",
            "maturity": "experimental"
        }

        cap = _deserialize_capability(entry, Path("test.yaml"))

        assert cap.id == "test.minimal"
        assert cap.name == "Minimal Test"
        assert cap.category == "horizontal"
        assert cap.description == "Minimal capability for testing"
        assert cap.maturity == "experimental"
        assert cap.metrics == {}
        assert cap.seeds == []
        assert cap.requirements == {}
        assert cap.activation == {}

    def test_deserialize_capability_full(self) -> None:
        """Test deserializing capability with all fields."""
        entry = {
            "id": "test.full",
            "name": "Full Test",
            "category": "vertical",
            "description": "Full capability for testing",
            "maturity": "stable",
            "metrics": {
                "accuracy": 0.95,
                "latency": None,
                "throughput": 100.5
            },
            "seeds": ["seed1", "seed2"],
            "requirements": {"python": ">=3.8"},
            "activation": {"enabled": True}
        }

        cap = _deserialize_capability(entry, Path("test.yaml"))

        assert cap.id == "test.full"
        assert cap.name == "Full Test"
        assert cap.category == "vertical"
        assert cap.maturity == "stable"
        assert cap.metrics == {"accuracy": 0.95, "latency": None, "throughput": 100.5}
        assert cap.seeds == ["seed1", "seed2"]
        assert cap.requirements == {"python": ">=3.8"}
        assert cap.activation == {"enabled": "True"}  # Values are converted to strings

    def test_deserialize_capability_invalid_metrics_value(self) -> None:
        """Test deserializing capability with invalid metrics value."""
        entry = {
            "id": "test.invalid",
            "name": "Invalid Metrics",
            "category": "horizontal",
            "description": "Test invalid metrics value",
            "maturity": "experimental",
            "metrics": {"invalid": "not_a_number"}
        }

        with pytest.raises(ValueError, match="Valor inválido em metrics"):
            _deserialize_capability(entry, Path("test.yaml"))

    def test_deserialize_capability_metrics_normalization(self) -> None:
        """Test that metrics values are properly normalized."""
        entry = {
            "id": "test.normalize",
            "name": "Normalize Test",
            "category": "horizontal",
            "description": "Test metrics normalization",
            "maturity": "experimental",
            "metrics": {
                "int_metric": 42,  # Should become float
                "float_metric": 3.14,
                "none_metric": None,
                "string_key": 95.5  # Numeric value with string key
            }
        }

        cap = _deserialize_capability(entry, Path("test.yaml"))

        assert cap.metrics["int_metric"] == 42.0  # int -> float
        assert cap.metrics["float_metric"] == 3.14
        assert cap.metrics["none_metric"] is None
        assert "string_key" in cap.metrics  # Key converted to string


class TestCapabilityIntegration:
    """Integration tests for capability system."""

    def test_full_capability_lifecycle(self) -> None:
        """Test complete capability lifecycle: creation, registry, updates, serialization."""
        # Create initial capability
        cap = Capability(
            id="test.lifecycle",
            name="Lifecycle Test",
            category="horizontal",
            description="Test complete lifecycle",
            maturity="experimental",
            metrics={"accuracy": 0.8}
        )

        # Add to registry
        raw_entries = {"test.lifecycle": {
            "id": "test.lifecycle",
            "name": "Lifecycle Test",
            "category": "horizontal",
            "description": "Test complete lifecycle",
            "maturity": "experimental",
            "metrics": {"accuracy": 0.8}
        }}

        registry = CapabilityRegistry([cap], raw_entries)

        # Update metrics and maturity
        registry.update_metrics({"test.lifecycle": {"accuracy": 0.9, "throughput": 150.0}})
        registry.update_maturity({"test.lifecycle": "beta"})

        # Serialize to YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            registry.to_yaml(temp_path)

            # Load back from YAML
            new_registry = CapabilityRegistry.from_yaml(temp_path)
            loaded_cap = new_registry.get("test.lifecycle")

            # Verify all changes persisted
            assert loaded_cap.maturity == "beta"
            assert loaded_cap.metrics["accuracy"] == 0.9
            assert loaded_cap.metrics["throughput"] == 150.0

        finally:
            Path(temp_path).unlink()

    def test_multiple_capabilities_management(self) -> None:
        """Test managing multiple capabilities in a registry."""
        capabilities = [
            Capability(id="test.1", name="Test 1", category="horizontal",
                      description="First test", maturity="experimental"),
            Capability(id="test.2", name="Test 2", category="vertical",
                      description="Second test", maturity="beta"),
            Capability(id="test.3", name="Test 3", category="horizontal",
                      description="Third test", maturity="stable")
        ]

        raw_entries = {cap.id: {"id": cap.id, "name": cap.name} for cap in capabilities}
        registry = CapabilityRegistry(capabilities, raw_entries)

        # Test bulk operations
        registry.update_metrics({
            "test.1": {"accuracy": 0.8},
            "test.2": {"latency": 100.0},
            "test.3": {"throughput": 200.0}
        })

        registry.update_maturity({
            "test.1": "beta",
            "test.2": "stable"
        })

        # Verify all capabilities updated correctly
        cap1 = registry.get("test.1")
        cap2 = registry.get("test.2")
        cap3 = registry.get("test.3")

        assert cap1.maturity == "beta"
        assert cap1.metrics["accuracy"] == 0.8

        assert cap2.maturity == "stable"
        assert cap2.metrics["latency"] == 100.0

        assert cap3.maturity == "stable"  # Unchanged
        assert cap3.metrics["throughput"] == 200.0

    def test_capability_registry_error_handling(self) -> None:
        """Test error handling in capability registry operations."""
        registry = CapabilityRegistry([], {})

        # Test operations on empty registry don't crash
        assert len(registry.list()) == 0
        assert registry.summary() == ""

        # Test updates on nonexistent capabilities are handled gracefully
        registry.update_metrics({"nonexistent": {"metric": 1.0}})
        registry.update_maturity({"nonexistent": "stable"})

        # Test serialization of empty registry
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            registry.to_yaml(temp_path)
            new_registry = CapabilityRegistry.from_yaml(temp_path)
            assert len(new_registry.list()) == 0
        finally:
            Path(temp_path).unlink()


class TestCapabilityEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_capability_with_special_characters(self) -> None:
        """Test capability with special characters in fields."""
        cap = Capability(
            id="test.special-chars_123",
            name="Test with spëcial çharß",
            category="horizontal",
            description="Capability with ümlauts and àccents",
            maturity="experimental"
        )

        registry = CapabilityRegistry([cap], {"test.special-chars_123": {"id": "test.special-chars_123"}})

        retrieved = registry.get("test.special-chars_123")
        assert retrieved.name == "Test with spëcial çharß"
        assert retrieved.description == "Capability with ümlauts and àccents"

    def test_large_capability_registry(self) -> None:
        """Test registry with many capabilities."""
        capabilities = []
        for i in range(100):
            cap = Capability(
                id=f"test.cap{i:03d}",
                name=f"Capability {i}",
                category="horizontal" if i % 2 == 0 else "vertical",
                description=f"Test capability {i}",
                maturity="experimental"
            )
            capabilities.append(cap)

        raw_entries = {cap.id: {"id": cap.id} for cap in capabilities}
        registry = CapabilityRegistry(capabilities, raw_entries)

        assert len(registry.list()) == 100

        # Test random access
        cap50 = registry.get("test.cap050")
        assert cap50.name == "Capability 50"
        assert cap50.category == "horizontal"  # i=50 is even, so horizontal

    def test_capability_metrics_extreme_values(self) -> None:
        """Test capability metrics with extreme values."""
        cap = Capability(
            id="test.extreme",
            name="Extreme Values Test",
            category="horizontal",
            description="Test extreme metric values",
            maturity="experimental",
            metrics={
                "zero": 0.0,
                "very_small": 1e-10,
                "very_large": 1e10,
                "negative": -100.5,
                "none": None
            }
        )

        registry = CapabilityRegistry([cap], {"test.extreme": {"id": "test.extreme"}})

        # Update with more extreme values
        registry.update_metrics({
            "test.extreme": {
                "inf": float('inf'),
                "neg_inf": float('-inf'),
                "nan": float('nan')
            }
        })

        # Verify extreme values are handled
        assert cap.metrics["zero"] == 0.0
        assert cap.metrics["very_small"] == 1e-10
        assert cap.metrics["very_large"] == 1e10
        assert cap.metrics["negative"] == -100.5
        assert cap.metrics["none"] is None