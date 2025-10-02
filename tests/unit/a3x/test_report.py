"""Comprehensive tests for the report module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from a3x.capabilities import Capability, CapabilityRegistry
from a3x.report import CapabilityUsage, generate_capability_report


class TestCapabilityUsage:
    """Test cases for the CapabilityUsage dataclass."""

    def test_capability_usage_creation_minimal(self) -> None:
        """Test creating capability usage with minimal fields."""
        usage = CapabilityUsage(
            capability_id="test.cap",
            name="Test Capability",
            category="horizontal"
        )

        assert usage.capability_id == "test.cap"
        assert usage.name == "Test Capability"
        assert usage.category == "horizontal"
        assert usage.runs == 0
        assert usage.completed_runs == 0

    def test_capability_usage_creation_with_runs(self) -> None:
        """Test creating capability usage with initial run counts."""
        usage = CapabilityUsage(
            capability_id="test.cap",
            name="Test Capability",
            category="vertical",
            runs=10,
            completed_runs=8
        )

        assert usage.capability_id == "test.cap"
        assert usage.name == "Test Capability"
        assert usage.category == "vertical"
        assert usage.runs == 10
        assert usage.completed_runs == 8

    def test_register_completed_run(self) -> None:
        """Test registering a completed run."""
        usage = CapabilityUsage(
            capability_id="test.cap",
            name="Test Capability",
            category="horizontal"
        )

        usage.register(True)

        assert usage.runs == 1
        assert usage.completed_runs == 1

    def test_register_failed_run(self) -> None:
        """Test registering a failed run."""
        usage = CapabilityUsage(
            capability_id="test.cap",
            name="Test Capability",
            category="horizontal"
        )

        usage.register(False)

        assert usage.runs == 1
        assert usage.completed_runs == 0

    def test_register_multiple_runs(self) -> None:
        """Test registering multiple runs with mixed results."""
        usage = CapabilityUsage(
            capability_id="test.cap",
            name="Test Capability",
            category="horizontal"
        )

        # Register various combinations
        usage.register(True)   # completed
        usage.register(False)  # failed
        usage.register(True)   # completed
        usage.register(True)   # completed
        usage.register(False)  # failed

        assert usage.runs == 5
        assert usage.completed_runs == 3

    def test_completion_rate_zero_runs(self) -> None:
        """Test completion rate with zero runs."""
        usage = CapabilityUsage(
            capability_id="test.cap",
            name="Test Capability",
            category="horizontal"
        )

        assert usage.completion_rate == 0.0

    def test_completion_rate_all_completed(self) -> None:
        """Test completion rate with all runs completed."""
        usage = CapabilityUsage(
            capability_id="test.cap",
            name="Test Capability",
            category="horizontal",
            runs=5,
            completed_runs=5
        )

        assert usage.completion_rate == 1.0

    def test_completion_rate_all_failed(self) -> None:
        """Test completion rate with all runs failed."""
        usage = CapabilityUsage(
            capability_id="test.cap",
            name="Test Capability",
            category="horizontal",
            runs=3,
            completed_runs=0
        )

        assert usage.completion_rate == 0.0

    def test_completion_rate_partial_completion(self) -> None:
        """Test completion rate with partial completion."""
        usage = CapabilityUsage(
            capability_id="test.cap",
            name="Test Capability",
            category="horizontal",
            runs=10,
            completed_runs=7
        )

        assert usage.completion_rate == 0.7

    def test_completion_rate_calculated_after_register(self) -> None:
        """Test that completion rate is calculated correctly after register calls."""
        usage = CapabilityUsage(
            capability_id="test.cap",
            name="Test Capability",
            category="horizontal"
        )

        # Initially 0 runs
        assert usage.completion_rate == 0.0

        # Register some runs
        usage.register(True)
        assert usage.completion_rate == 1.0

        usage.register(False)
        assert usage.completion_rate == 0.5

        usage.register(True)
        usage.register(True)
        assert usage.completion_rate == 0.75  # 3 out of 4 completed


class TestGenerateCapabilityReport:
    """Test cases for the generate_capability_report function."""

    def test_generate_report_no_capabilities_file(self) -> None:
        """Test generating report when capabilities file doesn't exist."""
        with patch("a3x.report.Path.exists", return_value=False):
            # Should not raise error, just return silently
            generate_capability_report(
                capabilities_path="nonexistent.yaml",
                output_path="output.md"
            )

    def test_generate_report_with_capabilities_no_usage(self) -> None:
        """Test generating report with capabilities but no usage data."""
        # Create temporary capabilities file
        capabilities_content = """
- id: "test.cap1"
  name: "Test Capability 1"
  category: "horizontal"
  description: "First test capability"
  maturity: "experimental"

- id: "test.cap2"
  name: "Test Capability 2"
  category: "vertical"
  description: "Second test capability"
  maturity: "beta"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                evaluations_log="nonexistent.jsonl",
                metrics_history="nonexistent.json",
                output_path=output_path
            )

            # Verify output file was created
            assert Path(output_path).exists()

            # Read and verify content
            content = Path(output_path).read_text(encoding='utf-8')

            assert "# SeedAI Capability Report" in content
            assert "Total evaluations: 0" in content
            assert "Overall completion rate: -" in content  # No evaluations
            assert "Test Capability 1" in content
            assert "Test Capability 2" in content
            assert "Nenhum uso registrado ainda." in content

        finally:
            Path(capabilities_path).unlink()
            Path(output_path).unlink()

    def test_generate_report_with_usage_data(self) -> None:
        """Test generating report with actual usage data."""
        # Create temporary capabilities file
        capabilities_content = """
- id: "test.cap1"
  name: "Test Capability 1"
  category: "horizontal"
  description: "First test capability"
  maturity: "experimental"

- id: "test.cap2"
  name: "Test Capability 2"
  category: "vertical"
  description: "Second test capability"
  maturity: "beta"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create temporary evaluations log
        evaluations_data = [
            {"capabilities": ["test.cap1"], "completed": True},
            {"capabilities": ["test.cap1", "test.cap2"], "completed": True},
            {"capabilities": ["test.cap2"], "completed": False},
            {"capabilities": ["test.cap1"], "completed": True}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for entry in evaluations_data:
                f.write(json.dumps(entry) + '\n')
            evaluations_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                evaluations_log=evaluations_path,
                output_path=output_path
            )

            # Read and verify content
            content = Path(output_path).read_text(encoding='utf-8')

            assert "Total evaluations: 4" in content
            assert "Completed: 3" in content
            assert "Overall completion rate: 75.00%" in content

            # Check capability usage
            assert "Test Capability 1 (test.cap1) | horizontal | 3 | 100.00%" in content
            assert "Test Capability 2 (test.cap2) | vertical | 2 | 50.00%" in content

        finally:
            Path(capabilities_path).unlink()
            Path(evaluations_path).unlink()
            Path(output_path).unlink()

    def test_generate_report_with_metrics_history(self) -> None:
        """Test generating report with metrics history."""
        # Create temporary capabilities file
        capabilities_content = """
- id: "test.cap1"
  name: "Test Capability 1"
  category: "horizontal"
  description: "First test capability"
  maturity: "experimental"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create temporary metrics history
        metrics_data = {
            "accuracy": [0.8, 0.85, 0.9, 0.95],
            "latency": [100.0, 95.0, 90.0, 85.0]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics_data, f)
            metrics_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                metrics_history=metrics_path,
                output_path=output_path
            )

            # Read and verify content
            content = Path(output_path).read_text(encoding='utf-8')

            # Check metrics summary
            assert "## Metrics Summary" in content
            assert "accuracy" in content
            assert "latency" in content
            assert "0.9500" in content  # Best accuracy
            assert "0.8500" in content  # Last latency

        finally:
            Path(capabilities_path).unlink()
            Path(metrics_path).unlink()
            Path(output_path).unlink()

    def test_generate_report_creates_output_directory(self) -> None:
        """Test that report generation creates output directory if needed."""
        # Create temporary capabilities file
        capabilities_content = """
- id: "test.cap1"
  name: "Test Capability 1"
  category: "horizontal"
  description: "First test capability"
  maturity: "experimental"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create nested output path
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_output = Path(temp_dir) / "nested" / "dir" / "report.md"

            generate_capability_report(
                capabilities_path=capabilities_path,
                output_path=nested_output
            )

            # Verify directory was created and file exists
            assert nested_output.exists()
            assert nested_output.parent.exists()

        Path(capabilities_path).unlink()

    def test_generate_report_with_empty_evaluations_log(self) -> None:
        """Test generating report with empty evaluations log."""
        # Create temporary capabilities file
        capabilities_content = """
- id: "test.cap1"
  name: "Test Capability 1"
  category: "horizontal"
  description: "First test capability"
  maturity: "experimental"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create empty evaluations log
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            evaluations_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                evaluations_log=evaluations_path,
                output_path=output_path
            )

            # Read and verify content
            content = Path(output_path).read_text(encoding='utf-8')

            assert "Total evaluations: 0" in content
            assert "Nenhum uso registrado ainda." in content

        finally:
            Path(capabilities_path).unlink()
            Path(evaluations_path).unlink()
            Path(output_path).unlink()

    def test_generate_report_with_malformed_evaluations_log(self) -> None:
        """Test generating report with malformed evaluations log."""
        # Create temporary capabilities file
        capabilities_content = """
- id: "test.cap1"
  name: "Test Capability 1"
  category: "horizontal"
  description: "First test capability"
  maturity: "experimental"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create evaluations log with malformed JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("invalid json line\n")
            f.write('{"capabilities": ["test.cap1"], "completed": true}\n')
            f.write("another invalid line\n")
            evaluations_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                evaluations_log=evaluations_path,
                output_path=output_path
            )

            # Should complete without error (malformed lines are skipped)
            content = Path(output_path).read_text(encoding='utf-8')
            assert "Total evaluations: 1" in content  # Only the valid line

        finally:
            Path(capabilities_path).unlink()
            Path(evaluations_path).unlink()
            Path(output_path).unlink()


class TestReportIntegration:
    """Integration tests for report functionality."""

    def test_full_report_generation_workflow(self) -> None:
        """Test complete report generation workflow."""
        # Create temporary capabilities file
        capabilities_content = """
- id: "planning.core"
  name: "Core Planning"
  category: "horizontal"
  description: "Core planning functionality"
  maturity: "stable"

- id: "execution.runtime"
  name: "Runtime Execution"
  category: "vertical"
  description: "Runtime execution engine"
  maturity: "beta"

- id: "memory.store"
  name: "Memory Store"
  category: "horizontal"
  description: "Memory storage system"
  maturity: "experimental"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create evaluations log with realistic data
        evaluations_data = [
            {"capabilities": ["planning.core"], "completed": True},
            {"capabilities": ["planning.core", "execution.runtime"], "completed": True},
            {"capabilities": ["execution.runtime"], "completed": False},
            {"capabilities": ["memory.store"], "completed": True},
            {"capabilities": ["planning.core", "memory.store"], "completed": True},
            {"capabilities": ["execution.runtime"], "completed": True}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for entry in evaluations_data:
                f.write(json.dumps(entry) + '\n')
            evaluations_path = f.name

        # Create metrics history
        metrics_data = {
            "actions_success_rate": [0.8, 0.85, 0.9, 0.95, 0.92],
            "apply_patch_success_rate": [0.9, 0.95, 0.98, 0.96],
            "tests_success_rate": [0.85, 0.88, 0.92, 0.94, 0.96]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics_data, f)
            metrics_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                evaluations_log=evaluations_path,
                metrics_history=metrics_path,
                output_path=output_path
            )

            # Read and verify complete report
            content = Path(output_path).read_text(encoding='utf-8')

            # Check header
            assert "# SeedAI Capability Report" in content

            # Check evaluation summary
            assert "Total evaluations: 6" in content
            assert "Completed: 5" in content
            assert "Overall completion rate: 83.33%" in content

            # Check capability usage
            assert "Core Planning (planning.core) | horizontal | 3 | 100.00%" in content
            assert "Runtime Execution (execution.runtime) | vertical | 3 | 66.67%" in content
            assert "Memory Store (memory.store) | horizontal | 2 | 100.00%" in content

            # Check metrics summary
            assert "## Metrics Summary" in content
            assert "actions_success_rate" in content
            assert "0.9500" in content  # Best value
            assert "0.9200" in content  # Last value

            # Check footer
            assert "Relatório gerado automaticamente pelo A3X SeedAI." in content

        finally:
            Path(capabilities_path).unlink()
            Path(evaluations_path).unlink()
            Path(metrics_path).unlink()
            Path(output_path).unlink()

    def test_report_with_registry_integration(self) -> None:
        """Test report generation with real capability registry."""
        # Create capabilities using the registry
        capabilities = [
            Capability(
                id="test.integration",
                name="Integration Test",
                category="horizontal",
                description="Test integration with registry",
                maturity="beta"
            )
        ]

        raw_entries = {
            "test.integration": {
                "id": "test.integration",
                "name": "Integration Test",
                "category": "horizontal",
                "description": "Test integration with registry",
                "maturity": "beta"
            }
        }

        registry = CapabilityRegistry(capabilities, raw_entries)

        # Create temporary capabilities file from registry
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            registry.to_yaml(f.name)
            capabilities_path = f.name

        # Create evaluations log
        evaluations_data = [
            {"capabilities": ["test.integration"], "completed": True},
            {"capabilities": ["test.integration"], "completed": True}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for entry in evaluations_data:
                f.write(json.dumps(entry) + '\n')
            evaluations_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                evaluations_log=evaluations_path,
                output_path=output_path
            )

            # Verify report was generated correctly
            content = Path(output_path).read_text(encoding='utf-8')
            assert "Integration Test (test.integration)" in content
            assert "Total evaluations: 2" in content
            assert "Overall completion rate: 100.00%" in content

        finally:
            Path(capabilities_path).unlink()
            Path(evaluations_path).unlink()
            Path(output_path).unlink()


class TestReportEdgeCases:
    """Test edge cases and error conditions."""

    def test_capability_usage_with_extreme_values(self) -> None:
        """Test capability usage with extreme run counts."""
        usage = CapabilityUsage(
            capability_id="test.extreme",
            name="Extreme Test",
            category="horizontal"
        )

        # Register extreme number of runs
        for _ in range(10000):
            usage.register(True)

        assert usage.runs == 10000
        assert usage.completed_runs == 10000
        assert usage.completion_rate == 1.0

    def test_report_with_very_long_capability_names(self) -> None:
        """Test report generation with very long capability names."""
        # Create capabilities file with long names
        capabilities_content = """
- id: "test.very.long.capability.id"
  name: "Very Long Capability Name That Exceeds Normal Length and Should Be Handled Properly"
  category: "horizontal"
  description: "A capability with an extremely long name for testing purposes"
  maturity: "experimental"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create evaluations log
        evaluations_data = [
            {"capabilities": ["test.very.long.capability.id"], "completed": True}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for entry in evaluations_data:
                f.write(json.dumps(entry) + '\n')
            evaluations_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                evaluations_log=evaluations_path,
                output_path=output_path
            )

            # Should handle long names without issues
            content = Path(output_path).read_text(encoding='utf-8')
            assert "Very Long Capability Name That Exceeds Normal Length" in content

        finally:
            Path(capabilities_path).unlink()
            Path(evaluations_path).unlink()
            Path(output_path).unlink()

    def test_report_with_unicode_content(self) -> None:
        """Test report generation with unicode content."""
        # Create capabilities file with unicode
        capabilities_content = """
- id: "test.unicode"
  name: "Unicode Capability çharactères spécîaux"
  category: "horizontal"
  description: "Capability with ümlauts and àccents for testing"
  maturity: "experimental"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create evaluations log
        evaluations_data = [
            {"capabilities": ["test.unicode"], "completed": True}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for entry in evaluations_data:
                f.write(json.dumps(entry) + '\n')
            evaluations_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                evaluations_log=evaluations_path,
                output_path=output_path
            )

            # Should preserve unicode characters
            content = Path(output_path).read_text(encoding='utf-8')
            assert "çharactères spécîaux" in content
            assert "ümlauts" in content
            assert "àccents" in content

        finally:
            Path(capabilities_path).unlink()
            Path(evaluations_path).unlink()
            Path(output_path).unlink()

    def test_report_with_missing_metrics_history(self) -> None:
        """Test report generation with missing metrics history file."""
        # Create temporary capabilities file
        capabilities_content = """
- id: "test.cap1"
  name: "Test Capability 1"
  category: "horizontal"
  description: "First test capability"
  maturity: "experimental"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                metrics_history="nonexistent.json",
                output_path=output_path
            )

            # Should handle missing metrics file gracefully
            content = Path(output_path).read_text(encoding='utf-8')
            assert "## Metrics Summary" in content
            assert "Nenhuma métrica registrada ainda." in content

        finally:
            Path(capabilities_path).unlink()
            Path(output_path).unlink()

    def test_report_with_empty_metrics_history(self) -> None:
        """Test report generation with empty metrics history."""
        # Create temporary capabilities file
        capabilities_content = """
- id: "test.cap1"
  name: "Test Capability 1"
  category: "horizontal"
  description: "First test capability"
  maturity: "experimental"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create empty metrics history
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            metrics_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                metrics_history=metrics_path,
                output_path=output_path
            )

            # Should handle empty metrics gracefully
            content = Path(output_path).read_text(encoding='utf-8')
            assert "Nenhuma métrica registrada ainda." in content

        finally:
            Path(capabilities_path).unlink()
            Path(metrics_path).unlink()
            Path(output_path).unlink()

    def test_report_with_malformed_metrics_history(self) -> None:
        """Test report generation with malformed metrics history."""
        # Create temporary capabilities file
        capabilities_content = """
- id: "test.cap1"
  name: "Test Capability 1"
  category: "horizontal"
  description: "First test capability"
  maturity: "experimental"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(capabilities_content)
            capabilities_path = f.name

        # Create malformed metrics history
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            metrics_path = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            generate_capability_report(
                capabilities_path=capabilities_path,
                metrics_history=metrics_path,
                output_path=output_path
            )

            # Should handle malformed metrics gracefully
            content = Path(output_path).read_text(encoding='utf-8')
            assert "Nenhuma métrica registrada ainda." in content

        finally:
            Path(capabilities_path).unlink()
            Path(metrics_path).unlink()
            Path(output_path).unlink()