# QWEN.md - Contexto Instrucional para Interações Futuras

## 🎯 Visão Geral do Projeto A3X

O **A3X** é um agente autônomo de codificação local baseado no conceito de **SeedAI** - uma inteligência artificial que evolui continuamente através de ciclos de auto-aprimoramento. O objetivo é criar um sistema que possa executar ciclos autônomos de edição → teste → correção em projetos locais, com foco em segurança, controlabilidade e crescimento contínuo.

### 🧠 Arquitetura Principal

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

## 🏗️ Estrutura do Código

```
A3X/
├── a3x/                    # Módulo principal
│   ├── cli.py             # Interface em linha de comando
│   ├── agent.py           # Orquestrador principal do agente
│   ├── executor.py        # Executor de ações (ApplyPatch, RunCommand, etc.)
│   ├── actions.py         # Definições de ações e observações
│   ├── config.py          # Carregamento e validação de configuração
│   ├── history.py         # Histórico de ações/observações com resumos
│   ├── patch.py           # Aplicação de diffs unificados
│   ├── llm.py            # Clientes LLM (OpenAI, Manual, etc.)
│   ├── autoeval.py       # Auto-avaliação e geração de seeds
│   ├── testgen.py        # Gerador de testes adaptativos
│   ├── report.py         # Relatórios de capacidades e métricas
│   ├── seeds.py          # Gerenciamento de backlog de seeds
│   ├── planning/         # Planejamento e missões
│   ├── memory/           # Memória semântica
│   └── meta_capabilities.py # Capacidades meta para auto-criação de habilidades
├── tests/                # Suite de testes
│   ├── unit/             # Testes unitários
│   │   └── a3x/          # Testes para cada módulo
│   └── generated/        # Testes gerados automaticamente
├── seed/                 # Artefatos SeedAI
│   ├── backlog.yaml      # Backlog de seeds propostas
│   ├── capabilities.yaml # Grafo de capacidades
│   ├── missions.yaml     # Missões e objetivos
│   ├── evaluations/      # Avaliações de execuções
│   ├── metrics/          # Métricas históricas
│   ├── reports/          # Relatórios de capacidades
│   └── memory/           # Memória semântica indexada
├── configs/              # Configurações do agente
├── docs/                 # Documentação
└── samples/              # Exemplos e demos
```

## 🧩 Componentes Principais

### 1. **Interface CLI (`a3x.cli`)**
- Aceita objetivo, arquivo de configuração e modo (dry-run, execução real)
- Comandos principais: `run`, `seed`, `autopilot`, `memory`

### 2. **Orquestrador do Agente (`a3x.agent`)**
- Implementa o loop de decisão/execução
- Gerencia limite de iterações, critérios de parada e coleta de métricas
- Coordena interação entre LLM, executor e histórico

### 3. **Cliente LLM (`a3x.llm`)**
- Abstração para modelos de linguagem
- Implementa clientes para OpenAI, Manual (roteiros YAML), OpenRouter
- Suporte a formatação de resposta JSON e testes com HTTP mockado

### 4. **Executor de Ações (`a3x.executor`)**
- Aplica ações: `ApplyPatch`, `RunCommand`, `ReadFile`, `WriteFile`, `Message`, `Finish`
- Controle de timeout e captura estruturada de stdout/stderr
- Análise de impacto pré-aplicação com validação de segurança

### 5. **Patch Manager (`a3x.patch`)**
- Aplica diffs unificados via `patch(1)` ou fallback em Python
- Validação de segurança e reversão automática em caso de falha

### 6. **Histórico (`a3x.history`)**
- Estruturas para logar ações/observações
- Gera resumos e snapshots do contexto para o LLM
- Truncamento por tokens aproximados para gerenciar contexto

### 7. **Auto-avaliação (`a3x.autoeval`)**
- Registra métricas de cada execução em `seed/evaluations/`
- Analisa código para identificar gaps de capacidades
- Gera seeds automáticas baseadas em métricas e desempenho

### 8. **Gerador de Testes (`a3x.testgen`)**
- Gera testes adaptativos em `tests/generated/`
- Garante crescimento contínuo das métricas rastreadas
- Cria testes que exigem evolução monotônica das métricas

### 9. **Planejamento e Missões (`a3x.planning`)**
- Sistema de missões multi-nível com milestones
- Planejador que gera seeds baseadas em objetivos e capacidades
- Armazenamento persistente de estado de missões

### 10. **Memória Semântica (`a3x.memory`)**
- Armazena resumos indexados dos runs em `seed/memory/memory.jsonl`
- Busca semântica por lembranças similares
- Integração com embeddings locais via sentence-transformers

### 11. **Capacidades Meta (`a3x.meta_capabilities`)**
- Sistema de auto-criação de habilidades
- Capacidades que permitem ao agente criar novas habilidades autonomamente
- Análise estática de código e geração de sugestões de otimização

## 🧪 Testes e Qualidade

### Estrutura de Testes
- **Testes Unitários**: Cobertura abrangente para cada módulo em `tests/unit/a3x/`
- **Testes Gerados**: Testes adaptativos que evoluem com o sistema em `tests/generated/`
- **Testes de Integração**: Verificação de fluxo completo do sistema

### Métricas de Qualidade
- **Cobertura de Testes**: 92+ testes passando com cobertura abrangente
- **Análise Estática**: Detecção de más práticas de código (números mágicos, variáveis globais, etc.)
- **Complexidade Ciclomática**: Monitoramento contínuo de complexidade de código

## 🔧 Comandos Principais

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

## 🌱 Conceito SeedAI

### Filosofia
O A3X implementa o ciclo de autoaprimoramento contínuo SeedAI:
1. **Edição incremental dirigida por diffs**
2. **Loop de auto-teste** com feedback estruturado
3. **Histórico compacto e contextualizado**
4. **Execução segura** com limites e isolamento

### Artefatos SeedAI
- **Logs & Métricas**: `seed/evaluations/run_evaluations.jsonl`
- **Testes Adaptativos**: `tests/generated/test_metrics_growth.py`
- **Relatórios**: `seed/reports/capability_report.md`
- **Capacidades**: `seed/capabilities.yaml` (grafo de habilidades)
- **Memória Semântica**: `seed/memory/memory.jsonl`
- **Missões**: `seed/missions.yaml` (objetivos multi-nível)
- **Meta Capabilities**: entries `meta.*` em capabilities.yaml

## 🚀 Roadmap de Evolução

### Fases Completas
1. ✅ **Análise Estática de Código**: Detecção de más práticas com análise AST
2. ✅ **Auto-otimização de Código**: Sugestões automáticas de refatoração
3. ✅ **Refatoração Inteligente**: Aplicação automática de melhorias de código
4. ✅ **Análise de Complexidade**: Monitoramento de complexidade ciclomática
5. ✅ **Rollback Automático**: Reversão inteligente de mudanças problemáticas

### Próximas Fases
1. 🔄 **Expansão Horizontal**: Aplicação do SeedAI a domínios além de desenvolvimento
2. 🔄 **Capacidades Meta**: Desenvolvimento de habilidades para auto-criação de novas habilidades
3. 🔄 **Aprendizado Transferível**: Capacidade de aplicar conhecimento entre domínios
4. 🔄 **Evolução Autodirigida**: Sistema que escolhe autonomamente quais capacidades desenvolver
5. 🔄 **Monetização**: Geração de receita através de valor entregue

## 📊 Métricas-Chave Monitoradas

### Métricas de Desempenho
- `apply_patch_success_rate`: Taxa de sucesso na aplicação de diffs
- `actions_success_rate`: Taxa de sucesso geral nas ações
- `tests_success_rate`: Taxa de sucesso nos testes automatizados
- `failure_rate`: Taxa de falhas nas execuções

### Métricas de Qualidade de Código
- `magic_numbers`: Contagem de números mágicos detectados
- `global_vars`: Contagem de variáveis globais
- `file_diversity`: Diversidade de tipos de arquivos modificados
- `complexity_score`: Pontuação de complexidade ciclomática

### Métricas de Aprendizado
- `capability_maturity`: Maturidade das diferentes capacidades
- `learning_curve`: Curva de aprendizado por domínio
- `skill_diversity`: Diversidade de habilidades desenvolvidas

## 🛡️ Segurança e Controle

### Políticas de Execução
- **Limites de Tempo**: Timeout configurável para comandos
- **Isolamento Opcional**: Sandboxing em containers Docker
- **Lista de Permissões/Negações**: Controle granular de comandos
- **Auditoria**: Log detalhado de todas as ações

### Validação de Segurança
- **Análise Estática Pré-execução**: Detecção de código perigoso
- **Verificação de Alinhamento**: Checagem de mudanças desalinhadas
- **Análise de Impacto**: Avaliação de consequências antes da aplicação
- **Rollback Automático**: Reversão inteligente de mudanças problemáticas

## 🧠 Conceitos-Chave

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

### Aprendizado Transferível
Capacidade de aplicar conhecimento e habilidades adquiridas em um domínio para resolver problemas em outros domínios.

## 📈 Estado Atual

O A3X está atualmente em estado de **protótipo avançado** com:

- ✅ **Loop autônomo completo**: Edição → Teste → Correção
- ✅ **Análise estática robusta**: Detecção de más práticas de código
- ✅ **Auto-otimização**: Sugestões e aplicação automática de melhorias
- ✅ **Refatoração inteligente**: Capacidade de melhorar automaticamente o código
- ✅ **Rollback automático**: Proteção contra degradação de qualidade
- ✅ **Sistema de seeds**: Geração automática de tarefas de melhoria
- ✅ **Testes abrangentes**: 92+ testes passando com cobertura completa
- ✅ **Segurança integrada**: Políticas rigorosas de segurança e controle

O sistema está pronto para evoluir para **domínios além do desenvolvimento de software** e implementar **capacidades meta de auto-criação de habilidades**.