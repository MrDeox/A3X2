from datetime import datetime, timezone

import pytest

from a3x.planner import PlannerThresholds
from a3x.seeds import AutoSeeder, Seed


@pytest.fixture
def auto_seeder():
    return AutoSeeder(thresholds=PlannerThresholds())


def test_monitor_and_seed_below_thresholds(auto_seeder):
    metrics = {
        "actions_success_rate": 0.7,  # below 0.8
        "tests_success_rate": 0.8,    # below 0.9
        "apply_patch_success_rate": 0.6  # below 0.8
    }
    seeds = auto_seeder.monitor_and_seed(metrics)
    assert len(seeds) == 3
    goals = [seed.goal for seed in seeds]
    assert "Refatorar executor para maior sucesso em ações" in goals
    assert "Expandir testgen para maior cobertura" in goals
    assert "Adicionar verificações de segurança ao patch.py" in goals


def test_monitor_and_seed_above_thresholds(auto_seeder):
    metrics = {
        "actions_success_rate": 0.9,  # above 0.8
        "tests_success_rate": 1.0,    # above 0.9
        "apply_patch_success_rate": 0.9  # above 0.8
    }
    seeds = auto_seeder.monitor_and_seed(metrics)
    assert len(seeds) == 0


def test_seed_attributes(auto_seeder):
    metrics = {"actions_success_rate": 0.5}
    seeds = auto_seeder.monitor_and_seed(metrics)
    if seeds:
        seed = seeds[0]
        assert isinstance(seed, Seed)
        assert seed.priority == "high"
        assert seed.type == "refactor"
        assert seed.config == "configs/sample.yaml"
        assert seed.max_steps == 8
        assert "created_at" in seed.metadata
        assert "triggered_by" in seed.metadata
        assert seed.metadata["triggered_by"] == "actions_success_rate"
        # ID should be unique with timestamp
        assert seed.id.startswith("auto.actions.success.rate.")
        timestamp = int(datetime.now(timezone.utc).timestamp())
        assert int(seed.id.split(".")[-1]) > (timestamp - 1)
