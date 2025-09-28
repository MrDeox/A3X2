# Roadmap AGI – A3X SeedAI Evolution

## 0. Baseline (Hoje)
- **Loop fechado**: `autopilot`, seeds, missões, memória semântica.
- **Domínio principal**: engenharia de software Python (diffs, testes, docs).
- **LLM**: modelos gratuitos via OpenRouter; nenhum hardware extra além deste Linux.

## 1. Currículos Verticais (Semanas 1-4)
- **Core.diffing → advanced**: currículo AST-aware (libcst, tree-sitter), seeds `meta.diffing.curriculum`.
- **Core.testing → established**: seeds de cobertura (`pytest --cov`), fuzzing leve, relatórios automáticos.
- **Governança mínima**: seeds de auditoria (detectar loops, gerar resumos diários em `seed/reports/`).

## 2. Planejamento Adaptativo (Semanas 5-8)
- **Auto-rotas**: seeds que editam `seed/goal_rotation.yaml` com base em métricas/memória.
- **Missões dinâmicas**: planner ajusta milestones, cria novas missões (lint, docs, novos domínios) automaticamente.
- **Memória ativa**: seeds consultam insights antes de agir; prompts se autoajustam usando memória relevante.

## 3. Expansão Horizontal (Semanas 9-16)
- **Novos domínios**: missões `horiz.web`, `horiz.data`, `horiz.infra` com currículos graduais usando libs leves (Flask, pandas, ansible/terraform local).
- **Toolkit**: seeds instalam/configuram ferramentas de linha de comando que cabem nesta máquina.
- **Observabilidade**: dashboards simples (Markdown/HTML) com KPIs de cada capability.

## 4. Meta-Raciocínio (Semanas 17-24)
- **Reflexão profunda**: seeds que analisam memória histórica e sugerem mudanças de estratégia.
- **Meta-capabilities 2.0**: seeds que criam novos meta seeds (ex.: currículo multi-agente).
- **Prompt policy tuning**: autopilot testa variações de prompts/configs e registra impacto.

## 5. Auto-Planejamento Estratégico (Meses 7-12)
- **Goal generation**: agente propõe objetivos macro (ex.: “aprender automação financeira”) baseado em lacunas/memória.
- **Economia de recursos**: seeds medem custo/tempo de execuções e priorizam tarefas com ROI melhor.
- **Segurança avançada**: monitoração de deriva, mecanismos de aprovação humana opcional.

## 6. Rumo à AGI (Ano 1+) – Pesquisa & Iteração
- **Multimodalidade leve**: integrar modelos/heurísticas para texto, código, dados tabulares; visão/voz somente se couber em recursos locais ou via APIs gratuitas.
- **Raciocínio simbólico**: incorporar frameworks de lógica/planejamento (ex.: pyDatalog, z3) como capabilities.
- **Meta-aprendizado**: permitir que o agente modifique seu próprio código/config com salvaguardas.
- **Governança humana**: definir políticas éticas, limites de atuação, mecanismos de auditoria contínua.

## 7. Uso Comercial & Autonomia Financeira
- **Identificação de oportunidades**: seeds que pesquisam problemas reais (bugs, freelas, automações) a partir da memória e web-crawlers controlados.
- **Prototipagem & entrega**: capabilities para gerar MVPs, documentação, scripts de deploy.
- **Ciclo receita-feedback**: conectar resultados (jobs entregues, feedback humano) como novas seeds/missões.

> Cada fase herda a anterior: avance apenas quando capacidades/métricas atingirem os thresholds definidos, mantendo logs, memória e auditoria ativos. Revisar o roadmap trimestralmente conforme aprendizados e constraints (modelos disponíveis, tempo, recursos locais).
