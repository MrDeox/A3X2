# 🧠 A3X SeedAI System Overview

## 🏗️ Arquitetura Principal

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

## 📁 Estrutura do Código

```
A3X/
├── a3x/                    # Módulo principal
│   ├── cli.py             # Interface em linha de comando
│   ├── agent.py           # Orquestrador principal do agente
│   ├── executor.py        # Executor de ações (ApplyPatch, RunCommand, etc.)
│   ├── actions.py        # Definições de ações e observações
│   ├── config.py         # Carregamento e validação de configuração
│   ├── history.py        # Histórico de ações/observações com resumos
│   ├── patch.py          # Aplicação de diffs unificados
│   ├── llm.py           # Clientes LLM (OpenAI, Manual, etc.)
│   ├── autoeval.py      # Auto-avaliação e geração de seeds
│   ├── testgen.py       # Gerador de testes adaptativos
│   ├── report.py        # Relatórios de capacidades e métricas
│   ├── seeds.py         # Gerenciamento de backlog de seeds
│   ├── planning/        # Planejamento e missões
│   ├── memory/          # Memória semântica
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
│   └── memory/          # Memória semântica indexada
├── configs/              # Configurações do agente
├── docs/                 # Documentação
└── samples/              # Exemplos e demos
```

## ⚙️ Componentes Principais

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

## 🧪 Pipeline de Execução

1. **Inicialização**
   - CLI carrega configuração
   - AgentOrchestrator é criado com configurações
   - LLMClient é instanciado (ManualLLMClient neste caso)

2. **Loop Principal**
   - Para cada iteração:
     a. LLMClient recebe histórico e retorna próxima ação
     b. Executor aplica ação e retorna resultado
     c. Histórico é atualizado com ação e observação
     d. Métricas são coletadas e avaliadas
     e. Auto-avaliação registra resultados

3. **Auto-avaliação**
   - Métricas são salvas em `seed/metrics/history.json`
   - Novas seeds são geradas com base em gaps identificados
   - Capacidades são atualizadas no grafo

4. **Geração de Testes Adaptativos**
   - Testes são gerados em `tests/generated/` para cobrir novas métricas
   - Garante que o sistema evolua de forma mensurável

## 📊 Métricas-Chave Monitoradas

- `actions_success_rate`: Taxa de sucesso das ações
- `apply_patch_success_rate`: Taxa de sucesso na aplicação de patches
- `tests_success_rate`: Taxa de sucesso dos testes
- `failure_rate`: Taxa de falhas
- `recursion_depth`: Profundidade de recursão
- `unique_commands`: Comandos únicos executados
- `unique_file_extensions`: Extensões de arquivos únicas modificadas

## 🌱 Sistema de Seeds

- Seeds são geradas automaticamente com base em gaps de capacidade
- Armazenadas em `seed/backlog.yaml`
- Executadas automaticamente em modo autopilot
- Cada seed representa uma oportunidade de melhoria

## 🧠 Sistema de Memória

- Histórico de execuções armazenado semanticamente
- Busca por similaridade usando embeddings
- Permite aprendizado transferível entre contextos