from pathlib import Path

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
