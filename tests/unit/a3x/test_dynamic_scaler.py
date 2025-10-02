"""Unit tests for DynamicScaler."""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
psutil_stub = types.ModuleType("psutil")
psutil_stub.cpu_percent = lambda interval=0: 0.0
psutil_stub.virtual_memory = lambda: types.SimpleNamespace(percent=0.0)
psutil_stub.disk_usage = lambda path: types.SimpleNamespace(percent=0.0)
psutil_stub.getloadavg = lambda: (0.0, 0.0, 0.0)
psutil_stub.cpu_count = lambda: 1
sys.modules.setdefault("psutil", psutil_stub)
from unittest.mock import Mock, patch

import pytest

from a3x.dynamic_scaler import DynamicScaler, ScalingDecision, integrate_dynamic_scaler


@pytest.fixture
def mock_config(tmp_path):
    config = Mock()
    config.workspace = Mock()
    config.workspace.root = tmp_path
    config.scaling = Mock(
        cpu_threshold=0.75,
        memory_threshold=0.8,
        max_recursion_adjust=3,
    )
    return config


@pytest.fixture
def dynamic_scaler(mock_config):
    return DynamicScaler(mock_config)


@patch("a3x.dynamic_scaler.psutil")
def test_monitor_resources(mock_psutil, dynamic_scaler):
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.virtual_memory.return_value = Mock(percent=60)
    mock_disk = Mock(used=1000000000, total=10000000000, percent=10.0)
    mock_psutil.disk_usage.return_value = mock_disk
    mock_psutil.getloadavg.return_value = (1.0, 2.0, 3.0)

    metrics = dynamic_scaler.monitor_resources()

    assert "cpu_percent" in metrics
    assert metrics["cpu_percent"] == 50.0
    assert metrics["memory_percent"] == 0.6
    assert metrics["disk_percent"] == 0.1
    assert "load_avg" in metrics


@patch("a3x.dynamic_scaler.psutil")
def test_make_scaling_decision_scale_down(mock_psutil, dynamic_scaler):
    mock_psutil.cpu_percent.return_value = 90.0
    mock_psutil.virtual_memory.return_value = Mock(percent=90)
    mock_psutil.disk_usage.return_value = Mock(used=1000000000, total=10000000000)
    mock_psutil.getloadavg.return_value = (1.0, 2.0, 3.0)

    metrics = {"cpu_percent": 0.9, "memory_percent": 0.9, "disk_percent": 0.1, "load_avg": 1.0}
    decision = dynamic_scaler.make_scaling_decision(metrics)

    assert decision.decision_type == "scale_down"
    assert "reduce_recursion_depth_or_pause" in decision.action_taken
    assert decision.confidence > 0.5
    assert dynamic_scaler.current_scaling_factor < 1.0


@patch("a3x.dynamic_scaler.psutil")
def test_make_scaling_decision_scale_up(mock_psutil, dynamic_scaler):
    mock_psutil.cpu_percent.return_value = 20.0
    mock_psutil.virtual_memory.return_value = Mock(percent=20)
    mock_psutil.disk_usage.return_value = Mock(used=1000000000, total=10000000000)
    mock_psutil.getloadavg.return_value = (0.5, 1.0, 1.5)

    metrics = {"cpu_percent": 0.2, "memory_percent": 0.2, "disk_percent": 0.1, "load_avg": 0.5}
    decision = dynamic_scaler.make_scaling_decision(metrics)

    assert decision.decision_type == "scale_up"
    assert "increase_complexity_if_possible" in decision.action_taken
    assert dynamic_scaler.current_scaling_factor > 1.0


def test_get_scaling_recommendation(dynamic_scaler):
    # Add a mock decision
    mock_decision = ScalingDecision(
        id="test", timestamp="2023-01-01", resource_metrics={}, decision_type="scale_down",
        action_taken="reduce", threshold_exceeded={}, confidence=0.8
    )
    dynamic_scaler.decision_history = [mock_decision]

    recommendation = dynamic_scaler.get_scaling_recommendation()

    assert "Scale down" in recommendation


def test_get_scaling_summary(dynamic_scaler):
    dynamic_scaler.decision_history = [
        ScalingDecision(id="1", timestamp="2023-01-01", resource_metrics={}, decision_type="scale_up", action_taken="up", threshold_exceeded={}, confidence=0.9),
        ScalingDecision(id="2", timestamp="2023-01-02", resource_metrics={}, decision_type="scale_down", action_taken="down", threshold_exceeded={}, confidence=0.7)
    ]
    dynamic_scaler.current_scaling_factor = 1.2

    summary = dynamic_scaler.get_scaling_summary()

    assert summary["history_length"] == 2
    assert abs(summary["avg_confidence"] - 0.8) < 0.01
    assert summary["scale_up_count"] == 1
    assert summary["current_scaling_factor"] == 1.2


def test_integration_function(mock_config):
    scaler = integrate_dynamic_scaler(mock_config)
    assert isinstance(scaler, DynamicScaler)


def test_reload_does_not_duplicate_history(mock_config):
    scaler = DynamicScaler(mock_config)
    decision = ScalingDecision(
        id="scale_20240101",
        timestamp="2024-01-01T00:00:00+00:00",
        resource_metrics={"cpu_percent": 0.5},
        decision_type="maintain",
        action_taken="monitor",
        threshold_exceeded={"cpu": False, "memory": False, "disk": False},
        confidence=0.5,
    )

    scaler.decision_history = [decision]
    scaler._save_scaling_decision(decision)
    scaler._save_scaling_history()

    reloaded_scaler = DynamicScaler(mock_config)
    assert len(reloaded_scaler.decision_history) == 1
    assert reloaded_scaler.decision_history[0].id == decision.id
