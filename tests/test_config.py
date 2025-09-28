from pathlib import Path
import yaml

import pytest

from a3x.config import load_config


def test_load_openrouter_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  type: openrouter
  model: "x-ai/grok-4-fast:free"
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
workspace:
  root: .
limits:
  max_iterations: 5
        """,
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    config = load_config(config_path)

    assert config.llm.type == "openrouter"
    assert config.llm.model == "x-ai/grok-4-fast:free"
    assert config.llm.base_url == "https://openrouter.ai/api/v1"
    assert config.workspace.root == tmp_path.resolve()
    assert config.limits.max_iterations == 5
    assert config.loop.auto_seed is False
    assert config.loop.seed_backlog == (tmp_path / "seed" / "backlog.yaml").resolve()
    assert config.loop.seed_config is None
    assert config.loop.stop_when_idle is True


def test_curriculum_thresholds():
    """Test that curriculum thresholds are set above 0.9."""
    curriculum_path = Path("configs/seed_self_improvement_curriculum.yaml")
    with open(curriculum_path, "r", encoding="utf-8") as f:
        curriculum = yaml.safe_load(f)
    
    # Check evaluation thresholds
    evaluation = curriculum["evaluation"]
    thresholds = evaluation["thresholds"]
    assert thresholds["vertical"] > 0.9
    assert thresholds["horizontal"] > 0.9
    assert thresholds["meta"] > 0.9
    
    # Check phase metrics_targets
    phases = curriculum["phases"]
    for phase in phases:
        for seed in phase.get("seeds", []):
            metrics_target = seed.get("metrics_target", {})
            if "actions_success_rate" in metrics_target:
                assert metrics_target["actions_success_rate"] > 0.9
            if "apply_patch_success_rate" in metrics_target:
                assert metrics_target["apply_patch_success_rate"] > 0.9
            if "self_patch_success_rate" in metrics_target:
                assert metrics_target["self_patch_success_rate"] > 0.9
            if "recursion_depth" in metrics_target:
                assert metrics_target["recursion_depth"] >= 5
