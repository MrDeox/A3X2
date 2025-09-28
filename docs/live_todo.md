# TODO Runtime – Ciclo Atual

## Fase 1 – Habituação
- [x] Painel de métricas automático pós-run (gerar `seed/reports/run_status.md`).
- [x] Seeds diagnósticas por threshold (ex.: `actions_success_rate` < 0.8).
- [x] Logging estruturado do daemon (registrar seed, status, tempo).

## Fase 2 – Plasticidade
- [x] Criar planner leve que prioriza seeds com base em métricas/gaps.
- [x] Registrar reflexão pós-run (report crítico) alimentando planner.
- [x] Executar seeds automaticamente após runs sem intervenção manual.
- [x] Atualizar missões SeedAI automaticamente com métricas agregadas.
- [ ] Implementar metas declarativas (ex.: `apply_patch_success_rate >= 0.9`).

## Fase 3 – Ampliação
- [x] Introduzir capability “core.tests” (rodar pytest real) com seeds/benchmarks.
- [x] Preparar seeds para lint (ruff/black).
- [ ] Preparar seeds para docs/web/dados conforme roadmap.
- [ ] Estender policies/sandbox.

## Fase 4 – Meta-aprendizado
- [ ] Iniciar memória longa (resumos/embeddings) para decisões futuras.
- [ ] Prompt/policy tuning automático baseado nos resultados.
- [ ] Ferramentas dinâmicas e primeiro protótipo de agentes especializados.
