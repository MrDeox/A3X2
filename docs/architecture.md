# Arquitetura Proposta para o Agente Autônomo A3X

Este documento detalha o desenho arquitetural inspirado pela pesquisa "Ferramenta Autônoma de Codificação Local – Pesquisa e Análise". O objetivo é construir um agente que consiga planejar, editar e testar código em um ambiente Linux local com o mínimo de intervenção humana.

## Princípios Norteadores

1. **Edição incremental segura** – utilizar diffs unificados como meio primário de alteração de arquivos, reduzindo risco de sobrescrever seções não relacionadas.
2. **Loop de auto-teste contínuo** – após cada alteração relevante, executar testes/linters e alimentar o resultado ao modelo.
3. **Autonomia controlada** – permitir autonomia prolongada (várias dezenas de iterações) mas com limites configuráveis de tempo, número de ações e recursos.
4. **Observabilidade e auditabilidade** – manter histórico claro de decisões, diffs aplicados e saídas de comando para auditoria e revisão humana.
5. **Extensibilidade** – código organizado em componentes substituíveis, habilitando integração com diferentes LLMs, runners e políticas.

## Diagrama de Alto Nível

```
┌───────────────────────┐                  ┌─────────────────────────┐
│       CLI / TUI       │  Objetivo        │      Configuração        │
└──────────┬────────────┘                  └──────────┬──────────────┘
           │                                        │
           v                                        v
┌───────────────────────┐        Decide       ┌─────────────────────────┐
│  AgentOrchestrator    ├────────────────────►│      LLM Client          │
└──────┬────────────────┘        Observa      └──────────┬──────────────┘
       │                                       Feedback │
       │                                                v
       │                               ┌──────────────────────────────────┐
       │                               │         História/Contexto         │
       │                               └──────────────────────────────────┘
       v
┌───────────────┐        ┌──────────────────┐        ┌───────────────────┐
│ PatchManager  │◄──────►│  CommandRunner   │        │  Policy Engine     │
└───────────────┘        └──────────────────┘        └───────────────────┘
       │                        │   ^                           │
       │                        │   │                           │
       v                        v   │                           v
┌───────────────┐        ┌──────────────┐           ┌─────────────────────┐
│ FileSystem    │        │ Test Harness │           │ Telemetria/Logs      │
└───────────────┘        └──────────────┘           └─────────────────────┘
```

## Componentes

### 1. CLI (`a3x.cli`)

- Fornece comandos `run` e `plan`.
- Recebe objetivo (`--goal`), caminho de config (`--config`), flags (`--dry-run`, `--max-steps`).
- Inicializa o `AgentOrchestrator` com dependências resolvidas.

### 2. Configuração (`a3x.config`)

Arquivo YAML valida e carrega:

- Dados do provedor de LLM (ex.: `type: openai`, `model: gpt-4o-mini`).
- Limites operacionais (`max_iterations`, `command_timeout`, `max_failures`).
- Policies (ex.: `deny_write_outside_workspace`, `allow_network: false`).
- Scripts de bootstrap (ex.: comandos antes da primeira ação).

### 3. LLM Client (`a3x.llm`)

Interface uniforme com métodos:

```python
class BaseLLMClient(ABC):
    def propose_action(self, state: AgentState) -> AgentAction:
        ...

    def summarize_history(self, history: AgentHistory) -> str:
        ...
```

Implementações previstas:
- `ManualLLMClient`: lê roteiro JSON/YAML; útil para testes sem modelo.
- `ChatLLMClient`: wrapper para OpenAI/Anthropic (usar asyncio + retry).
- `ProcessLLMClient`: executa binário externo (ex.: LM Studio) via stdin/stdout.

### 4. Histórico (`a3x.history`)

- Mantém lista de `Event` (ação + observação).
- Oferece `snapshot(max_tokens=4096)` que comprime entradas antigas em resumos.
- Permite exportar traces em JSON para auditoria.

### 5. Actions & Executor (`a3x.actions`, `a3x.executor`)

Ações suportadas inicialmente:

- `ApplyPatch(diff: str)` – aplicado via `patch`. Validação: dry-run antes de aplicar.
- `RunCommand(command: List[str], cwd: str)` – executa com timeout, `ulimit` opcional.
- `ReadFile(path: str)` – retorna conteúdo limitado.
- `WriteFile(path: str, content: str)` – fallback quando diff não é apropriado.
- `Finish(summary: str)` – encerra execução.

O `Executor` escolhe handler conforme tipo. Cada execução gera `Observation` com status (`success/failure`), saída textual e metadados (tempo, código de saída).

### 6. Test Harness (`a3x.testing`)

- Facilita execução de suites definidas no config (`commands: ["pytest", "ruff ."]`).
- Pode ser invocado automaticamente após ações que alteraram arquivos.

### 7. Policy Engine (`a3x.policy`)

- Regras configuráveis para validar ações antes da execução (ex.: bloqueio de `rm -rf /`).
- Integrado ao Executor; recusa ações violadoras e retorna observação de erro ao LLM.

### 8. Telemetria (`a3x.telemetry`)

- Logger estruturado (JSON ou texto) com níveis.
- Hooks para exporters futuros.

## Fluxo de Controle Detalhado

1. CLI coleta objetivo e config.
2. Config instancia `LLMClient` adequado.
3. Loop principal:
   1. `AgentState` compilado (objetivo, histórico resumido, status do workspace).
   2. `LLMClient.propose_action` gera `AgentAction`.
   3. `PolicyEngine.validate` avalia ação.
   4. `Executor.execute` aplica ação e devolve `Observation`.
   5. `History.append(action, observation)` atualiza estado.
   6. Critérios de parada avaliados.
4. Ao finalizar, relatório com diffs, logs e status de testes.

## Mapeamento para Requisitos da Pesquisa

| Requisito da pesquisa | Implementação A3X |
|-----------------------|-------------------|
| Edição via diff       | `ApplyPatch` + `patch` + fallback Python |
| Loop de auto-teste    | `TestHarness` executado após mutações    |
| Manutenção de contexto| `History.snapshot` + resumos             |
| Execução segura       | Timeout, policies, isolamento planejado  |
| Autonomia prolongada  | `max_iterations`, `max_failures`, logging|

## Extensões Futuras

- **Micro-agentes especializados**: submódulos com prompts/policies específicos.
- **Editor virtual**: implementar AST-aware editing usando `libcst` ou `tree-sitter`.
- **UI**: painel TUI similar ao Replit Agent 3 com streaming de ações.

## Referências

- Replit Agent 3 coverage – InfoQ (2024).
- OpenHands (ex-OpenDevin) – All Hands AI.
- SWE-Agent – Princeton/Stanford (SWE-Bench).
- bssw.io – boas práticas para agentes autônomos.
- Aider – edição de código orientada a diffs.

