# A3X - Ferramenta Autônoma de Codificação Local

## Visão Geral

O A3X é um agente autônomo de codificação local projetado para orquestrar um ciclo contínuo de edição → teste → correção em projetos locais, com foco em segurança, controlabilidade e crescimento contínuo. Inspirado em projetos como Replit Agent 3, OpenHands, SWE-Agent e GPT-Engineer, o A3X implementa o conceito de SeedAI - uma inteligência artificial que evolui continuamente através de ciclos de autoaprimoramento.

## Arquitetura

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

## Componentes Principais

### 1. Interface CLI (`a3x.cli`)
- Aceita objetivo, arquivo de configuração e modo (dry-run, execução real)
- Comandos principais: `run`, `seed`, `autopilot`, `memory`

### 2. Orquestrador do Agente (`a3x.agent`)
- Implementa o loop de decisão/execução
- Gerencia limite de iterações, critérios de parada e coleta de métricas
- Coordena interação entre LLM, executor e histórico

### 3. Cliente LLM (`a3x.llm`)
- Abstração para modelos de linguagem
- Implementa clientes para OpenAI, Manual (roteiros YAML), OpenRouter
- Cliente OpenRouter inclui fallback para Ollama quando necessário (inicializado sob demanda)
- Suporte a formatação de resposta JSON e testes com HTTP mockado

### 4. Executor de Ações (`a3x.executor`)
- Aplica ações: `ApplyPatch`, `RunCommand`, `ReadFile`, `WriteFile`, `Message`, `Finish`
- Controle de timeout e captura estruturada de stdout/stderr
- Análise de impacto pré-aplicação com validação de segurança

### 5. Patch Manager (`a3x.patch`)
- Aplica diffs unificados via `patch(1)` ou fallback em Python
- Validação de segurança e reversão automática em caso de falha

### 6. Histórico (`a3x.history`)
- Estruturas para logar ações/observações
- Gera resumos e snapshots do contexto para o LLM
- Truncamento por tokens aproximados para gerenciar contexto

### 7. Auto-avaliação (`a3x.autoeval`)
- Registra métricas de cada execução em `seed/evaluations/`
- Analisa código para identificar gaps de capacidades
- Gera seeds automáticas baseadas em métricas e desempenho

### 8. Gerador de Testes (`a3x.testgen`)
- Gera testes adaptativos em `tests/generated/`
- Garante crescimento contínuo das métricas rastreadas
- Cria testes que exigem evolução monotônica das métricas

### 9. Planejamento e Missões (`a3x.planning`)
- Sistema de missões multi-nível com milestones
- Planejador que gera seeds baseadas em objetivos e capacidades
- Armazenamento persistente de estado de missões

### 10. Memória Semântica (`a3x.memory`)
- Armazena resumos indexados dos runs em `seed/memory/memory.jsonl`
- Busca semântica por lembranças similares
- Integração com embeddings locais via sentence-transformers

### 11. Capacidades Meta (`a3x.meta_capabilities`)
- Sistema de auto-criação de habilidades
- Capacidades que permitem ao agente criar novas habilidades autonomamente
- Análise estática de código e geração de sugestões de otimização

## Comandos Básicos

### Execução Básica
```bash
# Instalação
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Execução com objetivo
a3x run --goal "Adicionar endpoint /health" --config configs/sample.yaml
```

### Execução de Seeds Autônomas
```bash
# Executa a próxima seed pendente
a3x seed run --config configs/seed_manual.yaml

# Loop autônomo (2 ciclos)
a3x autopilot --cycles 2 --goals seed/goal_rotation.yaml

# Loop contínuo
nohup ./scripts/autonomous_loop.sh > seed_watch.log 2>&1 &
```

## Artefatos SeedAI

- **Logs & Métricas**: `seed/evaluations/run_evaluations.jsonl` e `seed/metrics/history.json` agregam métricas por execução.
- **Testes Adaptativos**: `tests/generated/test_metrics_growth.py` é recriado automaticamente para exigir evolução monotônica das métricas rastreadas.
- **Relatórios**: `seed/reports/capability_report.md` resume uso de capacidades e melhores métricas.
- **Capacidades**: `seed/capabilities.yaml` serve como grafo de habilidades com seeds e métricas desejadas.
- **Memória Semântica**: `seed/memory/memory.jsonl` mantém resumos indexados dos runs; use `a3x memory search --query "texto"` para consultar.
- **Missões**: `seed/missions.yaml` descreve objetivos multi-nível; milestones incompletas geram seeds `mission.*` automaticamente.
- **Meta Capabilities**: entries `meta.*` em `seed/capabilities.yaml` disparam seeds de evolução quando requisitos de maturidade são atendidos (ex.: `meta.diffing.curriculum`).

## Filosofia SeedAI

O A3X implementa o ciclo de autoaprimoramento contínuo SeedAI:
1. **Edição incremental dirigida por diffs**: o agente gera patches unificados aplicados ao workspace.
2. **Loop de auto-teste**: comandos e suítes (ex.: `pytest`) são executados a cada iteração, e os resultados alimentam o próximo passo.
3. **Histórico compacto e contextualizado**: histórico de ações/observações com resumos para caber no contexto do modelo.
4. **Execução segura**: limites de tempo e isolamento opcional para comandos, mitigando riscos.

## Manifesto SeedAI

O projeto segue os princípios:
- **Human-first**: toda evolução deve aumentar a confiança e a utilidade para quem usa.
- **Seeds iterativas**: cada tarefa gera sementes de melhoria que alimentam o backlog evolutivo.
- **Aprendizado verificável**: todo aprimoramento passa por testes automatizados e revisão humana opcional.
- **Autonomia controlada**: o agente pode propor e executar mudanças, mas sempre respeitando políticas de segurança.
- **Memória auditável**: decisões, métricas e aprendizados ficam registrados em formato legível.

## Conceitos-Chave

### SeedAI (Inteligência Artificial Semeada)
Um sistema de IA que evolui continuamente através de ciclos de auto-aprimoramento, gerando automaticamente "seeds" (sementes) de melhoria que são cultivadas para expandir suas capacidades.

### Auto-modificação Segura
Capacidade do agente de modificar seu próprio código com salvaguardas robustas que previnem degradação de qualidade ou introdução de vulnerabilidades.

### Análise de Impacto Preditiva
Avaliação automática do impacto potencial de mudanças antes de sua aplicação, usando análise estática e métricas históricas.

### Refatoração Inteligente
Capacidade do sistema de identificar e aplicar automaticamente melhorias de código com base em padrões de qualidade e melhores práticas.

### Rollback Automático Inteligente
Sistema que reverte automaticamente mudanças problemáticas com base em métricas de qualidade e desempenho.

## Desenvolvimento e Contribuição

O A3X é licenciado sob a licença MIT e aceita contribuições via pull requests. O projeto segue práticas modernas de engenharia de software, com testes automatizados, CI/CD e documentação abrangente.