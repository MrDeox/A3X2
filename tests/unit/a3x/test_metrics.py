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
    rate_all_provided = compute_overall_success_rate(0.8, 0.9, 0.95, 0.85)
    assert rate_all_provided == 0.86  # Weighted sum rounded to 3 decimals

    # Some None, defaults to 0 or 1
    rate_with_defaults = compute_overall_success_rate(0.7, None, None, None)
    assert rate_with_defaults == 0.58

    # Edge case: all None
    rate_all_none = compute_overall_success_rate()
    assert rate_all_none == 0.3

    # Zero actions edge case
    rate_zero_actions_full_success = compute_overall_success_rate(0.0, 1.0, 1.0, 1.0)
    assert rate_zero_actions_full_success == 0.6

    rate_zero_actions_partial_success = compute_overall_success_rate(0.0, 0.9, 0.95, 0.85)
    assert rate_zero_actions_partial_success == 0.54

    # Lint or patch failure should not default to 1.0
    rate_zero_lint = compute_overall_success_rate(1.0, 1.0, 0.0, 1.0)
    assert rate_zero_lint == 0.85

    rate_zero_patch = compute_overall_success_rate(1.0, 1.0, 1.0, 0.0)
    assert rate_zero_patch == 0.85
