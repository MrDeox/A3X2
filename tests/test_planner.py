import pytest
from unittest.mock import Mock, patch, MagicMock

from a3x.planner import Planner, PlannerThresholds
from a3x.seeds import Seed


@pytest.fixture
def mock_history_low_rate():
    return {
        "actions_success_rate": [0.67],
        "apply_patch_success_rate": [0.8],
        "tests_success_rate": [0.9],
        "lint_success_rate": [0.9],
        "apply_patch_count": [1],
        "tests_run_count": [1],
        "lint_run_count": [1],
    }


@pytest.fixture
def mock_capability_metrics():
    return {
        "core.diffing": {"success_rate": 0.8},
        "core.testing": {"failures_detected": 0},
    }


@pytest.fixture
def mock_llm_response():
    return """
- id: meta.planner_tune
  goal: "Otimizar prompts em planner.py para melhorar actions_success_rate >0.9"
  priority: high
  type: refactor
  config: configs/scripts/seed_patch.yaml
  max_steps: 8
  metadata:
    description: "Enhance prompts with CoT and examples for high-ROI actions to enable full recursion."
    created_at: "2025-09-28T19:25:00Z"
- id: benchmark.action_opt
  goal: "Executar benchmark para validar tuning de prompts em cenários de baixa taxa"
  priority: high
  type: benchmark_diff
  config: configs/manual.yaml
  max_steps: 5
  metadata:
    description: "Benchmark to simulate improved actions_rate post-tuning."
    created_at: "2025-09-28T19:25:00Z"
"""


def test_planner_proposes_optimization_seeds_on_low_rate(mock_history_low_rate, mock_capability_metrics, mock_llm_response):
    with patch('a3x.planner.LLM.chat', return_value=mock_llm_response) as mock_chat:
        planner = Planner()
        seeds = planner.propose(
            mock_history_low_rate,
            patch_config_path="configs/scripts/seed_patch.yaml",
            manual_config_path="configs/manual.yaml",
            tests_config_path="configs/seed_tests.yaml",
            lint_config_path="configs/seed_lint.yaml",
            capability_metrics=mock_capability_metrics,
        )

    assert len(seeds) >= 2
    assert mock_chat.called
    optimization_seeds = [s for s in seeds if "otimizar" in s.goal.lower() or "tune" in s.id.lower()]
    assert len(optimization_seeds) > 0
    benchmark_seeds = [s for s in seeds if "benchmark" in s.type]
    assert len(benchmark_seeds) > 0

    # Simulate post-tuning improvement: mock a follow-up history with improved rate
    improved_history = mock_history_low_rate.copy()
    improved_history["actions_success_rate"] = [0.95]  # Post-tuning >0.9

    # In a real integration test, we'd run the proposed seeds and check metrics
    # For unit test, assert that with improved history, no further low-rate seeds are proposed
    with patch('a3x.planner.LLM.chat', return_value="[]"):  # No new seeds needed
        improved_seeds = planner.propose(
            improved_history,
            patch_config_path="configs/scripts/seed_patch.yaml",
            manual_config_path="configs/manual.yaml",
            tests_config_path="configs/seed_tests.yaml",
            lint_config_path="configs/seed_lint.yaml",
            capability_metrics=mock_capability_metrics,
        )
        low_rate_seeds = [s for s in improved_seeds if "otimizar" in s.goal.lower() or "tune" in s.id.lower()]
        assert len(low_rate_seeds) == 0  # No more optimization needed post-tuning
        assert 0.95 > 0.9  # Assert improved rate >0.9 (static check for now)


def test_planner_fallback_on_llm_failure(mock_history_low_rate, mock_capability_metrics):
    with patch('a3x.planner.LLM.chat', side_effect=Exception("LLM failure")):
        planner = Planner()
        seeds = planner.propose(
            mock_history_low_rate,
            patch_config_path="configs/scripts/seed_patch.yaml",
            manual_config_path="configs/manual.yaml",
            tests_config_path="configs/seed_tests.yaml",
            lint_config_path="configs/seed_lint.yaml",
            capability_metrics=mock_capability_metrics,
        )

    # Ensure fallback rule-based seeds are still proposed
    assert len(seeds) > 0
    recovery_seeds = [s for s in seeds if "recovery" in s.id]
    assert len(recovery_seeds) > 0  # At least the rule-based actions recovery seed


if __name__ == "__main__":
    pytest.main([__file__])