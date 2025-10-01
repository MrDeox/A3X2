# A3X Roadmap - Unified Path to SeedAI and AGI

This roadmap merges the AGI evolution path (from `roadmap_agi.md`) and SeedAI development phases (from `roadmap_seedai.md`) into a cohesive, phased plan. It prioritizes progressive autonomy, capability expansion, and self-improvement while incorporating metrics for measurable progress. Phases build cumulatively, with advancement gated by thresholds (e.g., success rates >0.8, test coverage >90%).

Timelines are estimates based on current baseline (Python software engineering domain, free LLMs via OpenRouter, local Linux resources). Review quarterly; adjust for learnings/constraints. Track via `seed/reports/` and `a3x autopilot --cycles N`.

## Vision

Evolve A3X from a code-focused agent to a plastic, self-directing SeedAI capable of AGI-like reasoning. Key principles:
- **Autonomy**: Closed-loop with seeds, memory, and adaptive planning.
- **Safety**: Policies, audits, human gates for high-risk actions.
- **Metrics-Driven**: Advance only when KPIs (e.g., `apply_patch_success_rate >= 0.9`) are met.
- **Resource Constraints**: No extra hardware; leverage free APIs and local tools.

## Phases

### Phase 0: Baseline (Current - Q4 2025)
Establish core loop stability.

**Key Milestones:**
- Closed autopilot loop with seeds, missions, semantic memory.
- Primary domain: Python engineering (diffs, tests, docs).
- LLM integration: Free models (e.g., Grok-4-fast).

**Timelines:** Immediate (already functional).
**Metrics:**
- Loop completion rate: 100% (no crashes).
- Basic capabilities: `actions_success_rate >= 0.7`.
- Audit: All changes logged in `seed/changes/`.

**Dependencies:** Stable configs (`sample.yaml`), basic tests passing.

### Phase 1: Habituation & Vertical Curricula (Q1 2026, Weeks 1-8)
Consolidate core and deepen vertical skills via curricula.

**Key Milestones:**
- Core capabilities: Advanced diffing (AST-aware with libcst/tree-sitter), established testing (pytest-cov, fuzzing, auto-reports).
- Governance: Audit seeds for loops, daily summaries in `seed/reports/`.
- Curricula: Encoded in `seed/curriculum.yaml` for gradual skill building (e.g., basic → advanced testing).

**Timelines:** 8 weeks; weekly seed runs.
**Metrics:**
- `apply_patch_success_rate >= 0.85` (core.diffing).
- `tests_success_rate >= 0.9`, coverage >= 80%.
- Backlog rotation: <= 5 pending seeds.
- Resource: Execution time < 5min/seed.

**Overlaps Resolved:** Combines SeedAI "Habituação" with AGI "Currículos Verticais"; focus on metrics panels and diagnostic seeds.

### Phase 2: Plasticity & Adaptive Planning (Q2 2026, Weeks 9-16)
Enable intelligent prioritization and dynamic adjustment.

**Key Milestones:**
- Auto-routes: Seeds edit `goal_rotation.yaml` based on metrics/memory.
- Dynamic missions: Planner adjusts milestones, auto-creates tasks (lint, docs, new domains).
- Active memory: Seeds query insights; prompts self-adjust.
- Reflection: Post-run critiques feed planner (failure reasons, next steps).

**Timelines:** 8 weeks; bi-weekly autopilot cycles.
**Metrics:**
- Planning accuracy: 80% tasks completed without replan.
- `actions_success_rate >= 0.85`.
- Memory reuse: >= 3 hits/run.
- Meta-seeds: Generate 1 new seed per 5 runs.

**Overlaps Resolved:** Merges AGI "Planejamento Adaptativo" with SeedAI "Plasticidade"; emphasize goal systems and thresholds (e.g., continue until `apply_patch_success_rate >= 0.9`).

### Phase 3: Ampliation - Horizontal & Vertical Expansion (Q3 2026, Weeks 17-32)
Broaden domains and toolkit.

**Key Milestones:**
- New domains: `horiz.web` (Flask), `horiz.data` (pandas), `horiz.infra` (local ansible/terraform).
- Toolkit: Seeds install/configure CLI tools fitting local machine.
- Curricula chaining: Basic to advanced per capability.
- Observability: Markdown/HTML dashboards for KPIs.
- Security: Sandbox (Docker/namespaces), resource limits, compliance checks.

**Timelines:** 16 weeks; monthly domain expansions.
**Metrics:**
- Domain coverage: 3+ domains with `success_rate >= 0.8`.
- Lint/format: `lint_success_rate >= 0.95`.
- Tool adoption: >= 2 new tools integrated.
- Safety: 0 high-risk violations (audited via `risk_log.md`).

**Overlaps Resolved:** Integrates AGI "Expansão Horizontal" with SeedAI "Ampliação"; add policies for sandboxing.

### Phase 4: Meta-Learning & Deep Reflection (Q4 2026, Months 7-9)
Enable self-modification and strategy evolution.

**Key Milestones:**
- Deep reflection: Seeds analyze historical memory, suggest strategy changes.
- Meta-capabilities 2.0: Seeds create new meta-seeds (e.g., multi-agent curricula).
- Prompt/policy tuning: Autopilot tests variations, logs impact.
- Long-term memory: Embeddings/resumes for future decisions.
- Multi-agent architecture: Coordinated planner/executor/critic/researcher.

**Timelines:** 12 weeks; continuous tuning.
**Metrics:**
- Reflection impact: Fitness delta >= 0.1 per quarter.
- Self-modify success: `self_modify_rate >= 0.7` (tests pass post-change).
- Policy improvement: 20% reduction in failures via tuning.

**Overlaps Resolved:** Combines AGI "Meta-Raciocínio" with SeedAI "Meta-aprendizado"; focus on tool discovery and ethical policies.

### Phase 5: Strategic Auto-Planning (Q1 2027, Months 10-12)
Achieve high-level autonomy.

**Key Milestones:**
- Goal generation: Agent proposes macro-objectives from gaps/memory (e.g., "Learn financial automation").
- Resource economy: Prioritize tasks by ROI (cost/time vs. impact).
- Advanced security: Drift monitoring, optional human approval.
- Multimodal light: Text/code/data heuristics; APIs for vision/voice if free.

**Timelines:** 12 weeks; strategic reviews monthly.
**Metrics:**
- Goal proposal quality: 80% accepted by human audit.
- ROI: Average fitness gain >= 0.15 per task.
- Security: 0 drifts detected (via memory audits).

### Phase 6: Towards AGI - Symbolic Reasoning & Research (2027+, Year 2)
Incorporate advanced reasoning.

**Key Milestones:**
- Symbolic reasoning: Integrate pyDatalog/z3 for logic/planning.
- Meta-learning: Agent modifies code/config with safeguards.
- Human governance: Ethical policies, continuous audit.
- Research mode: Controlled web-crawlers for problem discovery.

**Timelines:** Ongoing; annual milestones.
**Metrics:**
- Reasoning benchmarks: Solve 70% logic puzzles.
- Modification safety: 95% self-changes pass tests.
- Ethical compliance: 100% (audited).

### Phase 7: Commercial Use & Financial Autonomy (2028+)
Monetize capabilities.

**Key Milestones:**
- Opportunity ID: Seeds research real problems (bugs, freelancing).
- Prototyping: Generate MVPs, docs, deploy scripts.
- Revenue cycle: Connect outputs (delivered jobs) to new seeds/missions.

**Timelines:** Post-AGI stability; iterative.
**Metrics:**
- Delivery success: 80% client satisfaction.
- Revenue growth: Track via integrated metrics.

## Monitoring & Advancement

- **Gates:** Advance phases when 80% milestones met; use `a3x seed run` for validation.
- **Tools:** `a3x autopilot --goals goal_rotation.yaml` for rotation; dashboards in `seed/reports/`.
- **Risks:** Resource limits, model availability; mitigate with local fallbacks.
- **Next Steps:** Stabilize Phase 1 metrics; run `a3x run --config configs/seed_testing_curriculum.yaml`.

For details, see original roadmaps in git history. Contribute via `seed/backlog.yaml`.