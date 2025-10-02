"""Comprehensive tests for the policy module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from a3x.policy import PolicyOverrideManager


class TestPolicyOverrideManager:
    """Test cases for the PolicyOverrideManager class."""

    def test_manager_creation_default_path(self) -> None:
        """Test creating manager with default path."""
        with patch("a3x.policy.Path.exists", return_value=False):
            manager = PolicyOverrideManager()

            assert manager.path == Path("configs/policy_overrides.yaml")
            assert manager.data == {}

    def test_manager_creation_custom_path(self) -> None:
        """Test creating manager with custom path."""
        custom_path = Path("/custom/path/overrides.yaml")
        manager = PolicyOverrideManager(path=custom_path)

        assert manager.path == custom_path

    def test_manager_post_init_loads_data(self) -> None:
        """Test that post_init loads data from file."""
        yaml_content = """
agent:
  recursion_depth: 5
  max_failures: 3
"""

        with patch("a3x.policy.Path.exists", return_value=True):
            with patch("a3x.policy.Path.read_text", return_value=yaml_content):
                with patch("a3x.policy.yaml.safe_load", return_value={"agent": {"recursion_depth": 5, "max_failures": 3}}):
                    manager = PolicyOverrideManager()

                    assert manager.data == {"agent": {"recursion_depth": 5, "max_failures": 3}}

    def test_manager_post_init_handles_missing_file(self) -> None:
        """Test that post_init handles missing file gracefully."""
        with patch("a3x.policy.Path.exists", return_value=False):
            manager = PolicyOverrideManager()

            assert manager.data == {}

    def test_manager_post_init_handles_invalid_yaml(self) -> None:
        """Test that post_init handles invalid YAML gracefully."""
        with patch("a3x.policy.Path.exists", return_value=True):
            with patch("a3x.policy.Path.read_text", return_value="invalid: yaml: content: ["):
                with patch("a3x.policy.yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
                    manager = PolicyOverrideManager()

                    assert manager.data == {}

    def test_manager_post_init_handles_non_dict_yaml(self) -> None:
        """Test that post_init handles non-dict YAML content."""
        with patch("a3x.policy.Path.exists", return_value=True):
            with patch("a3x.policy.Path.read_text", return_value="just a string"):
                with patch("a3x.policy.yaml.safe_load", return_value="just a string"):
                    manager = PolicyOverrideManager()

                    assert manager.data == {}

    def test_apply_to_agent_recursion_depth_valid(self) -> None:
        """Test applying valid recursion depth override to agent."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 7}}

        # Mock agent
        agent = MagicMock()
        agent.recursion_depth = 5

        manager.apply_to_agent(agent)

        assert agent.recursion_depth == 7

    def test_apply_to_agent_recursion_depth_invalid_type(self) -> None:
        """Test applying invalid recursion depth type (should not crash)."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": "invalid"}}

        agent = MagicMock()
        agent.recursion_depth = 5

        # Should not raise exception and should not change value
        manager.apply_to_agent(agent)
        assert agent.recursion_depth == 5

    def test_apply_to_agent_recursion_depth_invalid_value(self) -> None:
        """Test applying invalid recursion depth value (should not crash)."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": -1}}

        agent = MagicMock()
        agent.recursion_depth = 5

        # Should not raise exception and should not change value
        manager.apply_to_agent(agent)
        assert agent.recursion_depth == 5

    def test_apply_to_agent_max_failures_valid(self) -> None:
        """Test applying valid max failures override to agent."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"max_failures": 10}}

        # Mock agent with config
        agent = MagicMock()
        agent.config.limits.max_failures = 5

        manager.apply_to_agent(agent)

        assert agent.config.limits.max_failures == 10

    def test_apply_to_agent_max_failures_invalid_type(self) -> None:
        """Test applying invalid max failures type (should not crash)."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"max_failures": "invalid"}}

        agent = MagicMock()
        agent.config.limits.max_failures = 5

        # Should not raise exception and should not change value
        manager.apply_to_agent(agent)
        assert agent.config.limits.max_failures == 5

    def test_apply_to_agent_max_failures_invalid_value(self) -> None:
        """Test applying invalid max failures value (should not crash)."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"max_failures": 0}}

        agent = MagicMock()
        agent.config.limits.max_failures = 5

        # Should not raise exception and should not change value
        manager.apply_to_agent(agent)
        assert agent.config.limits.max_failures == 5

    def test_apply_to_agent_no_overrides(self) -> None:
        """Test applying to agent with no overrides set."""
        manager = PolicyOverrideManager()
        manager.data = {}

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 3

        # Should not change anything
        manager.apply_to_agent(agent)
        assert agent.recursion_depth == 5
        assert agent.config.limits.max_failures == 3

    def test_apply_to_agent_partial_overrides(self) -> None:
        """Test applying to agent with only some overrides set."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 8}}  # Only recursion_depth, no max_failures

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 3

        manager.apply_to_agent(agent)

        assert agent.recursion_depth == 8  # Changed
        assert agent.config.limits.max_failures == 3  # Unchanged

    def test_update_from_report_reduce_recursion_depth(self) -> None:
        """Test updating from report that recommends reducing recursion depth."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 5, "max_failures": 3}}

        # Mock agent
        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 3

        # Mock report with recursion depth recommendation
        report = MagicMock()
        report.recommendations = ["Reduzir profundidade recursiva para melhorar estabilidade"]

        manager.update_from_report(report, agent)

        # Should reduce recursion depth by 1 (max with 3)
        assert manager.data["agent"]["recursion_depth"] == 4
        assert manager.data["agent"]["max_failures"] == 3  # Unchanged

    def test_update_from_report_increase_supervision(self) -> None:
        """Test updating from report that recommends increasing supervision."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 5, "max_failures": 3}}

        # Mock agent
        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 3

        # Mock report with supervision recommendation
        report = MagicMock()
        report.recommendations = ["Aumentar supervisão para reduzir erros"]

        manager.update_from_report(report, agent)

        # Should reduce max_failures by 1 (max with 3)
        assert manager.data["agent"]["recursion_depth"] == 5  # Unchanged
        assert manager.data["agent"]["max_failures"] == 2

    def test_update_from_report_multiple_recommendations(self) -> None:
        """Test updating from report with multiple recommendations."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 5, "max_failures": 3}}

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 3

        report = MagicMock()
        report.recommendations = [
            "Reduzir profundidade recursiva para melhorar estabilidade",
            "Aumentar supervisão para reduzir erros",
            "Irrelevant recommendation"
        ]

        manager.update_from_report(report, agent)

        # Both values should be reduced
        assert manager.data["agent"]["recursion_depth"] == 4
        assert manager.data["agent"]["max_failures"] == 2

    def test_update_from_report_no_relevant_recommendations(self) -> None:
        """Test updating from report with no relevant recommendations."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 5, "max_failures": 3}}

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 3

        report = MagicMock()
        report.recommendations = ["Irrelevant recommendation", "Another irrelevant one"]

        manager.update_from_report(report, agent)

        # Values should remain unchanged
        assert manager.data["agent"]["recursion_depth"] == 5
        assert manager.data["agent"]["max_failures"] == 3

    def test_update_from_report_no_existing_overrides(self) -> None:
        """Test updating from report when no overrides exist yet."""
        manager = PolicyOverrideManager()
        manager.data = {}

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 3

        report = MagicMock()
        report.recommendations = ["Reduzir profundidade recursiva para melhorar estabilidade"]

        manager.update_from_report(report, agent)

        # Should create new overrides
        assert manager.data["agent"]["recursion_depth"] == 4
        assert manager.data["agent"]["max_failures"] == 3  # Unchanged (no recommendation)

    def test_update_from_report_below_minimum_values(self) -> None:
        """Test that updates don't go below minimum values."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 3, "max_failures": 3}}  # Already at minimum

        agent = MagicMock()
        agent.recursion_depth = 3
        agent.config.limits.max_failures = 3

        report = MagicMock()
        report.recommendations = [
            "Reduzir profundidade recursiva para melhorar estabilidade",
            "Aumentar supervisão para reduzir erros"
        ]

        manager.update_from_report(report, agent)

        # Should stay at minimum values
        assert manager.data["agent"]["recursion_depth"] == 3
        assert manager.data["agent"]["max_failures"] == 3

    def test_update_from_report_saves_changes(self) -> None:
        """Test that updates are saved to file."""
        manager = PolicyOverrideManager()

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 3

        report = MagicMock()
        report.recommendations = ["Reduzir profundidade recursiva para melhorar estabilidade"]

        with patch.object(manager, '_save') as mock_save:
            manager.update_from_report(report, agent)
            mock_save.assert_called_once()

    def test_update_from_report_applies_to_agent(self) -> None:
        """Test that updates are applied to agent after saving."""
        manager = PolicyOverrideManager()

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 3

        report = MagicMock()
        report.recommendations = ["Reduzir profundidade recursiva para melhorar estabilidade"]

        manager.update_from_report(report, agent)

        # Should apply the new values to agent
        assert agent.recursion_depth == 4


class TestPolicyManagerPersistence:
    """Test cases for policy manager persistence."""

    def test_save_creates_directory(self) -> None:
        """Test that save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "path" / "overrides.yaml"
            manager = PolicyOverrideManager(path=nested_path)
            manager.data = {"agent": {"recursion_depth": 5}}

            manager._save()

            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_save_writes_correct_yaml(self) -> None:
        """Test that save writes correct YAML format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)

        try:
            manager = PolicyOverrideManager(path=temp_path)
            manager.data = {
                "agent": {
                    "recursion_depth": 5,
                    "max_failures": 3
                }
            }

            manager._save()

            # Read back and verify
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            data = yaml.safe_load(content)
            assert data == {"agent": {"recursion_depth": 5, "max_failures": 3}}

        finally:
            temp_path.unlink()

    def test_save_with_sort_keys_false(self) -> None:
        """Test that save preserves key order (sort_keys=False)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)

        try:
            manager = PolicyOverrideManager(path=temp_path)
            manager.data = {
                "z_last": "value_z",
                "a_first": "value_a",
                "m_middle": "value_m"
            }

            manager._save()

            # Read back and verify order is preserved
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # YAML dumper with sort_keys=False should preserve insertion order
            lines = content.strip().split('\n')
            assert any('z_last:' in line for line in lines)
            assert any('a_first:' in line for line in lines)
            assert any('m_middle:' in line for line in lines)

        finally:
            temp_path.unlink()

    def test_load_corrupted_file_returns_empty_dict(self) -> None:
        """Test that loading corrupted YAML file returns empty dict."""
        with patch("a3x.policy.Path.exists", return_value=True):
            with patch("a3x.policy.Path.read_text", return_value="corrupted: yaml: [unclosed"):
                with patch("a3x.policy.yaml.safe_load", side_effect=yaml.YAMLError("Corrupted")):
                    manager = PolicyOverrideManager()

                    assert manager.data == {}


class TestPolicyManagerIntegration:
    """Integration tests for policy manager."""

    def test_full_policy_lifecycle(self) -> None:
        """Test complete policy lifecycle: load, update, apply, save."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            initial_content = """
agent:
  recursion_depth: 5
  max_failures: 3
"""
            f.write(initial_content)
            temp_path = Path(f.name)

        try:
            # Load existing policy
            manager = PolicyOverrideManager(path=temp_path)

            # Verify initial load
            assert manager.data["agent"]["recursion_depth"] == 5
            assert manager.data["agent"]["max_failures"] == 3

            # Mock agent
            agent = MagicMock()
            agent.recursion_depth = 5
            agent.config.limits.max_failures = 3

            # Apply to agent
            manager.apply_to_agent(agent)
            assert agent.recursion_depth == 5
            assert agent.config.limits.max_failures == 3

            # Update from report
            report = MagicMock()
            report.recommendations = [
                "Reduzir profundidade recursiva para melhorar estabilidade",
                "Aumentar supervisão para reduzir erros"
            ]

            manager.update_from_report(report, agent)

            # Verify updates
            assert manager.data["agent"]["recursion_depth"] == 4
            assert manager.data["agent"]["max_failures"] == 2

            # Verify agent was updated
            assert agent.recursion_depth == 4
            assert agent.config.limits.max_failures == 2

            # Verify file was saved
            with open(temp_path, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            saved_data = yaml.safe_load(saved_content)
            assert saved_data["agent"]["recursion_depth"] == 4
            assert saved_data["agent"]["max_failures"] == 2

        finally:
            temp_path.unlink()

    def test_policy_manager_with_real_agent_mock(self) -> None:
        """Test policy manager with realistic agent mock."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 6, "max_failures": 5}}

        # Create realistic agent mock
        from unittest.mock import MagicMock

        # Mock the config structure
        config_mock = MagicMock()
        limits_mock = MagicMock()
        limits_mock.max_failures = 3
        config_mock.limits = limits_mock

        agent_mock = MagicMock()
        agent_mock.recursion_depth = 4
        agent_mock.config = config_mock

        # Apply policies
        manager.apply_to_agent(agent_mock)

        # Verify policies were applied
        assert agent_mock.recursion_depth == 6
        assert agent_mock.config.limits.max_failures == 5

    def test_multiple_reports_accumulate_changes(self) -> None:
        """Test that multiple reports accumulate policy changes."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 5, "max_failures": 5}}

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 5

        # First report
        report1 = MagicMock()
        report1.recommendations = ["Reduzir profundidade recursiva para melhorar estabilidade"]

        manager.update_from_report(report1, agent)

        assert manager.data["agent"]["recursion_depth"] == 4
        assert manager.data["agent"]["max_failures"] == 5

        # Second report
        report2 = MagicMock()
        report2.recommendations = ["Aumentar supervisão para reduzir erros"]

        manager.update_from_report(report2, agent)

        assert manager.data["agent"]["recursion_depth"] == 4
        assert manager.data["agent"]["max_failures"] == 4

        # Third report with both recommendations
        report3 = MagicMock()
        report3.recommendations = [
            "Reduzir profundidade recursiva para melhorar estabilidade",
            "Aumentar supervisão para reduzir erros"
        ]

        manager.update_from_report(report3, agent)

        assert manager.data["agent"]["recursion_depth"] == 3
        assert manager.data["agent"]["max_failures"] == 3


class TestPolicyManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_extreme_policy_values(self) -> None:
        """Test policy manager with extreme values."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 1000, "max_failures": 1000}}

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 5

        manager.apply_to_agent(agent)

        # Should apply extreme values
        assert agent.recursion_depth == 1000
        assert agent.config.limits.max_failures == 1000

    def test_concurrent_policy_updates(self) -> None:
        """Test handling concurrent policy updates."""
        manager = PolicyOverrideManager()
        manager.data = {"agent": {"recursion_depth": 5, "max_failures": 5}}

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 5

        # Simulate concurrent updates by calling update multiple times quickly
        reports = []
        for i in range(5):
            report = MagicMock()
            report.recommendations = ["Reduzir profundidade recursiva para melhorar estabilidade"]
            reports.append(report)

        for report in reports:
            manager.update_from_report(report, agent)

        # Should have reduced recursion depth 5 times (but capped at 3)
        assert manager.data["agent"]["recursion_depth"] == 3
        assert agent.recursion_depth == 3

    def test_policy_with_unicode_content(self) -> None:
        """Test policy manager with unicode content in recommendations."""
        manager = PolicyOverrideManager()

        agent = MagicMock()
        agent.recursion_depth = 5

        report = MagicMock()
        report.recommendations = [
            "Reduzir profundidade recursiva para melhorar estabilidade",
            "Recomendação com çharactères spécîaux",
            "Unicode: 测试 試験 試験"
        ]

        # Should handle unicode without issues
        manager.update_from_report(report, agent)

        assert manager.data["agent"]["recursion_depth"] == 4

    def test_policy_manager_with_empty_recommendations(self) -> None:
        """Test policy manager with empty recommendations list."""
        manager = PolicyOverrideManager()

        agent = MagicMock()
        agent.recursion_depth = 5
        agent.config.limits.max_failures = 5

        report = MagicMock()
        report.recommendations = []

        # Should not change anything
        original_data = manager.data.copy()
        manager.update_from_report(report, agent)
        assert manager.data == original_data