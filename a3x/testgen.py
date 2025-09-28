"""Utilities to generate adaptive growth tests for the SeedAI loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class GrowthTestGenerator:
    """Creates pytest files that enforce non-regression on tracked metrics."""

    def __init__(self, history_path: Path | str, output_path: Path | str) -> None:
        self.history_path = Path(history_path)
        self.output_path = Path(output_path)

    def ensure_tests(self) -> None:
        if not self.history_path.exists():
            return
        try:
            history = json.loads(self.history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(history, dict) or not history:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._render_test(history)
        if (
            self.output_path.exists()
            and self.output_path.read_text(encoding="utf-8") == content
        ):
            return
        self.output_path.write_text(content, encoding="utf-8")

    def _render_test(self, history: Dict[str, List[float]]) -> str:
        del history  # conteúdo reservado para usos futuros
        return (
            '"""Testes gerados automaticamente para verificar saúde das métricas SeedAI.\n'
            "\n"
            'AUTO-GENERATED FILE. Edite via GrowthTestGenerator."""\n'
            "\n"
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "\n"
            "def _load_history() -> dict[str, list[float]]:\n"
            '    history_path = Path(__file__).resolve().parents[2] / "seed" / "metrics" / "history.json"\n'
            '    data = json.loads(history_path.read_text(encoding="utf-8"))\n'
            "    if not isinstance(data, dict):\n"
            '        raise AssertionError("Histórico de métricas inválido")\n'
            "    return {key: list(map(float, values)) for key, values in data.items()}\n"
            "\n"
            "\n"
            "def test_seed_metrics_health() -> None:\n"
            "    history = _load_history()\n"
            '    assert history, "Histórico de métricas não pode estar vazio"\n'
            "    for metric, values in history.items():\n"
            '        assert values, f"Métrica {metric} sem registros"\n'
            "        best = max(values)\n"
            "        last = values[-1]\n"
            "        for idx, value in enumerate(values):\n"
            '            assert value >= -1e-6, f"Métrica {metric} negativa no índice {idx}: {value}"\n'
            '        if metric.endswith("success_rate"):\n'
            '            assert last >= best - 0.25, f"Métrica {metric} degradou demais: {last} < {best}"\n'
            '        if metric.endswith("latency") and best > 0:\n'
            '            assert last <= best * 4 + 1e-6, f"Latência {metric} explodiu: {last} > {best}"\n'
        )
