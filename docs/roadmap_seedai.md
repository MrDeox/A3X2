# Roadmap SeedAI – A3X

> Metas progressivas para evoluir o A3X rumo a uma SeedAI autônoma, plástica e expansiva.

## Visão Geral

1. **Habituação** – consolidar núcleo (ciclo consistente, métricas claras, auditoria).
2. **Plasticidade** – planejamento e análise inteligentes priorizando capacidades/metrics.
3. **Ampliação** – expandir capacidades horizontais e verticais em novos domínios.
4. **Meta-aprendizado** – memória longa, ajuste de políticas, criação de ferramentas, rumo a emergências AGI.

---

## Fase 1 – Habituação (Núcleo Sólido)

- Painel de métricas (CLI/relatório) pós-run com tendências de `apply_patch_success_rate`, `actions_success_rate`, etc.
- Seeds diagnósticas por métrica: thresholds (ex.: actions_success_rate < 0.8) disparando seeds corretivas.
- Daemon com logging estruturado (`seed/daemon.log`) indicando cada seed executada e resultado.
- Auditoria ops: confirmar diffs em `seed/changes`, backlog girando sem pendências.

## Fase 2 – Plasticidade (Escolha Inteligente)

- Planner leve (`a3x/planner.py`) que lê métricas/backlog e gera prioridades.
- Reflexão/critic: relatório pós-run alimentando planner (motivos de falha, próximos passos).
- Sistema de metas (ex.: `apply_patch_success_rate ≥ 0.9`) – seeds continuam até meta atingida.

## Fase 3 – Ampliação (Horizontal & Vertical)

- Novas capabilities: testes Python reais, lint/format, docs avançadas, exemplo web/dados.
- [x] `core.tests`: seeds para executar Pytest em `samples/core_tests_demo` e monitorar `tests_success_rate`.
- [x] `core.lint`: seeds para rodar ruff/black e acompanhar `lint_success_rate`.
- Currículos: seeds encadeadas do básico ao avançado em cada capability.
- Segurança/policies: sandbox (Docker/user namespaces), limites de recursos, compliance.

## Fase 4 – Meta-aprendizado (Rumo a AGI)

- Memória longa com resumos/embeddings para decisões futuras.
- Ajuste de prompts/políticas automático (feedback → tuning).
- Descoberta de ferramentas: seeds que instalam/integraram libs conforme necessidade.
- Arquitetura multi-agente (planner, executor, crítico, pesquisador) coordenada por políticas.

---

## Lista TODO Dinâmica (docs/live_todo.md)

Principal controladora de tarefas ativas. Atualizar conforme itens acima entram em execução.
