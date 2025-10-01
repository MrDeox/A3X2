# A3X TODO - Current Priorities

This file tracks active tasks for A3X development. Outdated items (e.g., initial OpenRouter integration, basic CLI setup) have been removed. For full backlog, see [seed/backlog.yaml](seed/backlog.yaml). Report new issues via `a3x seed run` or add to backlog.

## High Priority (Immediate)

- [ ] Implement sandbox for executor (Docker/user namespaces) to enhance security. Link: [seed/backlog.yaml - security sandbox](seed/backlog.yaml)
- [ ] Add TUI/observability mode for real-time iteration monitoring. Link: [docs/architecture.md#observability](docs/architecture.md)
- [ ] Auto-generate benchmark tasks per capability and integrate into pipeline. Link: [seed/curriculum.yaml](seed/curriculum.yaml)

## Medium Priority

- [ ] Introduce token/cost limits per execution in config. Link: [configs/sample.yaml - limits](configs/sample.yaml)
- [ ] Enhance GrowthTestGenerator with dynamic thresholds and capability-specific cases. Link: [a3x/testgen.py](a3x/testgen.py)
- [ ] Expand PolicyEngine for glob-based rules and dangerous command lists. Link: [a3x/policy.py](a3x/policy.py)

## Future Priorities

- [ ] Automate test suite execution on triggers (e.g., post-pip install). Link: [tests/integration/](tests/integration/)
- [ ] Implement multi-domain expansion (web, data, infra) with light libs. Link: [docs/roadmap.md#phase-3](docs/roadmap.md)
- [ ] Add meta-learning for prompt/policy auto-tuning. Link: [seed/memory/](seed/memory/)

For progress tracking, run `a3x autopilot --cycles 3` and review `seed/reports/`. Validate changes with `pytest -q` and `ruff check .`.
