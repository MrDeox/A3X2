"""Centralized metrics calculations for A3X."""



def get_latest_metric(history: dict[str, list[float]], metric: str) -> float | None:
    """Retrieve the latest value for a given metric from history."""
    values = history.get(metric, [])
    return values[-1] if values else None


def compute_overall_success_rate(
    actions_rate: float | None = None,
    tests_rate: float | None = None,
    lint_rate: float | None = None,
    patch_rate: float | None = None,
) -> float:
    """
    Compute weighted overall success rate.
    
    Weights:
    - actions_success: 0.4
    - tests_success: 0.3
    - lint_success: 0.15
    - patch_success: 0.15
    """
    rates = {
        "actions_success": actions_rate or 0.0,
        "tests_success": tests_rate or 0.0,
        "lint_success": lint_rate if lint_rate is not None else 1.0,
        "patch_success": patch_rate if patch_rate is not None else 1.0,
    }
    weights = {
        "actions_success": 0.4,
        "tests_success": 0.3,
        "lint_success": 0.15,
        "patch_success": 0.15,
    }
    total = sum(weights[k] * v for k, v in rates.items())
    return round(total, 3)
