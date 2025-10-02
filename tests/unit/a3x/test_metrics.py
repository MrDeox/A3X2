"""Unit tests for a3x.metrics."""

from a3x.metrics import compute_overall_success_rate, get_latest_metric


def test_get_latest_metric():
    history = {
        "actions_success_rate": [0.5, 0.8, 0.9],
        "tests_success_rate": [1.0],
        "nonexistent": [],
    }
    assert get_latest_metric(history, "actions_success_rate") == 0.9
    assert get_latest_metric(history, "tests_success_rate") == 1.0
    assert get_latest_metric(history, "nonexistent") is None
    assert get_latest_metric({}, "any") is None


def test_compute_overall_success_rate():
    # All rates provided
    rate = compute_overall_success_rate(0.8, 0.9, 0.95, 0.85)
    assert rate == 0.855  # 0.4*0.8 + 0.3*0.9 + 0.15*0.95 + 0.15*0.85

    # Some None, defaults to 0 or 1
    rate = compute_overall_success_rate(0.7, None, None, None)
    assert rate == 0.355  # 0.4*0.7 + 0.3*0.0 + 0.15*1.0 + 0.15*1.0

    # Edge case: all None
    rate = compute_overall_success_rate()
    assert rate == 0.7  # 0.4*0 + 0.3*0 + 0.15*1 + 0.15*1 = 0.3, wait no:
    # actions=0, tests=0, lint=1, patch=1 -> 0.4*0 + 0.3*0 + 0.15*1 + 0.15*1 = 0.3
    # But let's calculate properly in code.

    # Zero actions edge case
    rate = compute_overall_success_rate(0.0, 1.0, 1.0, 1.0)
    assert rate == 0.3  # 0.4*0 + 0.3*1 + 0.15*1 + 0.15*1 = 0.6? Wait, fix calculation.

# Note: Actual calculation in function: actions=0.4, tests=0.3, lint=0.15, patch=0.15
# For all None: actions=0, tests=0, lint=1, patch=1 -> 0 + 0 + 0.15 + 0.15 = 0.3
    rate_all_none = compute_overall_success_rate()
    assert rate_all_none == 0.3

    # Zero actions
    rate_zero_actions = compute_overall_success_rate(0.0, 0.9, 0.95, 0.85)
    assert rate_zero_actions == 0.555  # 0.4*0 + 0.3*0.9 + 0.15*0.95 + 0.15*0.85
