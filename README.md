# A3X – Ferramenta Autônoma de Codificação Local

A3X é um esqueleto de ferramenta CLI desenhada para orquestrar um agente de codificação local inspirado no Replit Agent 3 e em iniciativas open-source como OpenHands, SWE-Agent e GPT-Engineer. O objetivo é oferecer uma base extensível para executar um ciclo autônomo de edição → teste → correção em projetos locais, com foco em segurança e controlabilidade.

## Motivação

Com base na pesquisa "Ferramenta Autônoma de Codificação Local – Pesquisa e Análise", buscamos consolidar práticas modernas:

- **Edição incremental dirigida por diffs:** o agente gera patches unificados aplicados ao workspace.
- **Loop de auto-teste:** comandos e suítes (ex.: `pytest`) são executados a cada iteração, e os resultados alimentam o próximo passo.
- **Histórico compacto e contextualizado:** histórico de ações/observações com resumos para caber no contexto do modelo.
- **Execução segura:** limites de tempo e isolamento opcional para comandos, mitigando riscos.

## Visão Geral da Arquitetura

```
+----------------------+        +---------------------+
| CLI (a3x)            | -----> | AgentOrchestrator   |
+----------------------+        +----------+----------+
                                          |
                                          v
                      +-----------------------------------------+
                      | Loop Autônomo                          |
                      | 1. LLM decide próxima ação              |
                      | 2. Executor aplica ação                 |
                      | 3. Histórico é atualizado               |
                      +----------------+------------------------+
                                       |
          +----------------------------+----------------------------+
          |                             |                            |
          v                             v                            v
+--------------------+      +-----------------------+    +----------------------+
| LLMClient          |      | Workspace/Patcheador  |    | CommandRunner        |
| (OpenAI/local/etc) |      | Aplica diffs unificados|    | Executa testes/CLI   |
+--------------------+      +-----------------------+    +----------------------+
```

### Componentes

- `a3x.cli`: Interface em linha de comando; aceita objetivo, arquivo de configuração e modo (dry-run, execução real, etc.).
- `a3x.agent`: Implementa o loop de decisão/execução, incluindo limite de iterações, critérios de parada e coleta de métricas.
- `a3x.llm`: Abstração para o modelo de linguagem. Inclui `ManualLLMClient` (roteiros YAML) e `OpenRouterLLMClient`, pronto para Grok 4 Fast ou outros modelos disponíveis na OpenRouter.
- `a3x.executor`: Aplicação de ações (`ApplyPatch`, `RunCommand`, `ReadFile`, `Message`) com controle de timeout e captura de stdout/stderr.
- `a3x.patch`: Utilitário que aplica diffs unificados via `patch(1)` ou usando fallback em Python.
- `a3x.history`: Estruturas para logar ações/observações, gerar resumos e snapshots do contexto para o LLM.
- `a3x.config`: Carrega arquivo YAML declarando ferramentas habilitadas, limites de recursos e prompts iniciais.
- `docs/seed_manifesto.md`: Manifesto SeedAI descrevendo o ciclo de autoaprimoramento contínuo (horizontal e vertical).
- `seed/capabilities.yaml`: Grafo de capacidades monitoradas com seeds propostas e métricas alvo.
- `a3x.autoeval`: Esqueleto de autoavaliação que registra métricas de cada execução em `seed/evaluations/`.
- `a3x.testgen`: Gera testes adaptativos em `tests/generated/` para garantir crescimento contínuo das métricas.
- `a3x.report`: Consolida relatórios de uso de capacidades e métricas em `seed/reports/`.

## Fluxo Operacional

1. **Objetivo inicial**: fornecido via CLI, incorporado ao contexto inicial.
2. **Planejamento**: o `LLMClient` gera a primeira ação (ex.: criar arquivo inicial).
3. **Execução**: a ação é despachada para o `Executor` (rodar comando, aplicar patch, etc.).
4. **Observação**: stdout/stderr e metadados são agregados ao `History`.
5. **Iteração**: o contexto resumido é enviado de volta ao LLM, que decide a próxima ação.
6. **Encerramento**: o agente pode emitir `Finish` (com relatório final) ou ser interrompido por limite de iterações/tempo.

## Estado Atual

- Estrutura de código inicial com interfaces e implementações básicas.
- Diff aplicado via utilitário `patch` com fallback puro em Python.
- Executor com timeout configurável e captura estruturada de resultados.
- Histórico com resumos e truncamento por tokens aproximados.
- Cliente `OpenRouterLLMClient` com suporte a Grok 4 Fast (ou outro modelo), response format em JSON e testes unitários com HTTP mockado.
- Manifesto SeedAI e ciclo de autoaprimoramento documentado.
- Autoavaliação básica persistindo avaliações em JSONL para alimentar seeds futuras.
- Métricas históricas, testes gerados automaticamente e relatório de capacidades atualizados a cada execução.

## Roadmap Sugerido

1. **Integração com provedores LLM**: implementar clientes para OpenAI, Anthropic ou modelos hospedados localmente (ex.: LM Studio, vLLM).
2. **Sandbox de execução**: rodar comandos dentro de contêiner Docker ou user namespaces para isolar efeitos colaterais.
3. **Gestão de memória avançada**: resumos hierárquicos e persistência em disco.
4. **Suporte a múltiplos projetos**: permitir workspaces paralelos e plugins específicos de linguagem.
5. **Observabilidade**: integrar logging estruturado, métricas Prometheus e painel TUI opcional.
6. **Growth SeedAI**: automatizar coleta de métricas, cataloga capacidades e criar seeds de evolução recorrentes.

## Uso Básico

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Defina a variável OPENROUTER_API_KEY (ex.: via arquivo .env)
a3x run --goal "Adicionar endpoint /health" --config configs/sample.yaml
```

O arquivo `configs/sample.yaml` está pronto para usar o modelo Grok 4 Fast via OpenRouter (`llm.model: "x-ai/grok-4-fast:free"`). Ajuste o campo `llm.model` conforme o slug desejado listado em <https://openrouter.ai/models>. Caso queira testar o ciclo sem chamadas externas, utilize `configs/manual.yaml`, que trabalha com scripts previamente definidos em `configs/scripts/demo_plan.yaml`.

### Execução de Seeds Autônomas

```bash
# Executa a próxima seed pendente do backlog (default: seed/backlog.yaml)
a3x seed run --config configs/seed_manual.yaml

# Ou via módulo Python (útil para pipelines CI)
python -m a3x.seed_runner --backlog seed/backlog.yaml --config configs/seed_manual.yaml

# Loop autônomo (ex.: 2 ciclos com rotação definida em seed/goal_rotation.yaml)
a3x autopilot --cycles 2 --goals seed/goal_rotation.yaml

# Loop contínuo (script helper)
nohup ./scripts/autonomous_loop.sh > seed_watch.log 2>&1 &
```

As seeds vivem em `seed/backlog.yaml` com prioridade, configuração, status e histórico. Ao rodar, o seed runner marca o item como `in_progress`, executa o agente e atualiza para `completed`/`failed` com notas e timestamp, sustentando o ciclo de autoaprimoramento.

## Artefatos SeedAI

- **Logs & Métricas**: `seed/evaluations/run_evaluations.jsonl` e `seed/metrics/history.json` agregam métricas por execução.
- **Testes Adaptativos**: `tests/generated/test_metrics_growth.py` é recriado automaticamente para exigir evolução monotônica das métricas rastreadas.
- **Relatórios**: `seed/reports/capability_report.md` resume uso de capacidades e melhores métricas.
- **Capacidades**: `seed/capabilities.yaml` serve como grafo de habilidades com seeds e métricas desejadas.
- **Memória Semântica**: `seed/memory/memory.jsonl` mantém resumos indexados dos runs; use `a3x memory search --query "texto"` para consultar.
- **Missões**: `seed/missions.yaml` descreve objetivos multi-nível; milestones incompletas geram seeds `mission.*` automaticamente.
- **Meta Capabilities**: entries `meta.*` em `seed/capabilities.yaml` disparam seeds de evolução quando requisitos de maturidade são atendidos (ex.: `meta.diffing.curriculum`).
- Para habilitar embeddings locais, instale `sentence-transformers` (o modelo padrão `all-MiniLM-L6-v2` roda em CPU).

## Licença

MIT
