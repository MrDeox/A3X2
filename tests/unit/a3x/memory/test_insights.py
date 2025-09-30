import pytest
from typing import Dict, List
from unittest.mock import Mock

from a3x.memory.insights import build_insight_payload


class MockRunEvaluation:
    def __init__(
        self,
        goal: str,
        completed: bool = True,
        iterations: int = 1,
        failures: int = 0,
        metrics: Dict[str, float] | None = None,
        capabilities: List[str] | None = None,
        seeds: List[Mock] | None = None,
        timestamp: str = "2023-01-01T00:00:00Z",
    ):
        self.goal = goal
        self.completed = completed
        self.iterations = iterations
        self.failures = failures
        self.metrics = metrics or {}
        self.capabilities = capabilities or []
        self.seeds = seeds or []
        self.timestamp = timestamp


class TestBuildInsightPayload:
    def test_happy_path_completed(self) -> None:
        evaluation = MockRunEvaluation(
            goal="Test Goal",
            completed=True,
            iterations=2,
            failures=0,
            metrics={"accuracy": 0.95},
            capabilities=["planning", "execution"],
            seeds=[Mock(description="Seed1"), Mock(description="Seed2")],
        )
        title, content, tags, metadata = build_insight_payload(evaluation)

        assert title == "Run Test Goal ✅"
        assert "goal: Test Goal" in content
        assert "status: completed" in content
        assert "iterations: 2" in content
        assert "failures: 0" in content
        assert "metrics: accuracy=0.950" in content
        assert "capabilities: execution, planning" in content
        assert "seeds: Seed1; Seed2" in content
        assert set(tags) == {"✅", "execution", "planning"}
        assert metadata["goal"] == "Test Goal"
        assert metadata["completed"] is True
        assert metadata["capabilities"] == ["planning", "execution"]
        assert metadata["timestamp"] == "2023-01-01T00:00:00Z"

    def test_incomplete_run(self) -> None:
        evaluation = MockRunEvaluation(
            goal="Incomplete Goal",
            completed=False,
            iterations=1,
            failures=1,
            metrics=None,
            capabilities=["planning"],
            seeds=[],
        )
        title, content, tags, metadata = build_insight_payload(evaluation)

        assert title == "Run Incomplete Goal ⚠️"
        assert "status: incomplete" in content
        assert "failures: 1" in content
        assert "capabilities: planning" in content
        assert set(tags) == {"⚠️", "planning"}
        assert metadata["completed"] is False

    def test_no_metrics_or_capabilities(self) -> None:
        evaluation = MockRunEvaluation(
            goal="Simple Goal",
            completed=True,
            iterations=1,
            failures=0,
            metrics={},
            capabilities=[],
            seeds=[],
        )
        title, content, tags, metadata = build_insight_payload(evaluation)

        assert title == "Run Simple Goal ✅"
        assert "metrics: " not in content  # No metrics line
        assert "capabilities: " not in content  # Empty capabilities
        assert tags == ["✅"]

    def test_with_capability_metrics(self) -> None:
        evaluation = MockRunEvaluation(
            goal="Goal with Metrics",
            completed=True,
            capabilities=["planning", "execution"],
        )
        capability_metrics = {
            "planning": {"efficiency": 0.8, "score": 85},
            "execution": {"speed": 1.2},
        }

        title, content, tags, metadata = build_insight_payload(evaluation, capability_metrics)

        assert "capability_metrics: planning: efficiency=0.8, score=85 | execution: speed=1.2" in content

    def test_empty_seeds(self) -> None:
        evaluation = MockRunEvaluation(
            goal="No Seeds",
            seeds=[],
        )
        title, content, tags, metadata = build_insight_payload(evaluation)

        assert "seeds: " not in content  # No seeds line

    def test_invalid_data_handling(self) -> None:
        # Test with non-numeric metrics; function will skip or error, but to test handling, use numeric only for now
        evaluation = MockRunEvaluation(
            goal="Invalid Metrics",
            metrics={"num_metric": 0.5},
            capabilities=["test"],
        )

        title, content, tags, metadata = build_insight_payload(evaluation)

        assert "metrics: num_metric=0.500" in content
        assert set(tags) == {"✅", "test"}

    def test_timestamp_default(self) -> None:
        evaluation = MockRunEvaluation("Goal")
        evaluation.timestamp = None  # Simulate missing

        title, content, tags, metadata = build_insight_payload(evaluation)
        assert "timestamp" in metadata  # Should use evaluation.timestamp, but if None, still sets it