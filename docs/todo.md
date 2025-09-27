# TODO – A3X Autônomo

## Prioridade Imediata
- [x] Implementar `OpenRouterLLMClient` usando HTTP (`httpx`), suportando modelos configuráveis e leitura da API key via variável de ambiente.
- [x] Estender `configs/sample.yaml` para aceitar `type: openrouter`, `model`, `base_url` e `api_key_env`, garantindo compatibilidade retroativa com o modo manual.
- [x] Ajustar `a3x.cli`/`AgentOrchestrator` para validar que o cliente escolhido possui credenciais e tratar respostas de erro da OpenRouter com mensagens úteis.
- [x] Definir prompts base (sistema/usuário) para o loop de ações (planejamento geral + foco em diffs/tests).
- [x] Escrever testes unitários para o cliente OpenRouter (mockando HTTP) e para o caminho de configuração correspondente.
- [ ] Projetar backlog SeedAI com seeds automáticas e fluxo de seleção/execução.
- [ ] Implementar seeds de benchmark que forçam criação de diffs reais (destravar `apply_patch_success_rate`).
- [ ] Registrar métricas de latência/retries do LLM no autoavaliação.

## Prioridade Média
- [x] Adicionar mecanismo simples de retry/backoff para requisições ao LLM.
- [ ] Criar comando CLI `a3x plan` ou flag `--dry-run` que apenas valida config/credenciais.
- [ ] Introduzir opções de limite de tokens/custos por execução.
- [ ] Registrar logs estruturados com níveis (INFO/DEBUG) e permitir `--verbose` na CLI.
- [ ] Evoluir GrowthTestGenerator para utilizar thresholds dinâmicos e geração de casos específicos por capability.

## Prioridade Futuras
- [ ] Implementar sandbox opcional (Docker ou user namespaces) no `ActionExecutor`.
- [ ] Expandir `PolicyEngine` para regras baseadas em padrões glob e lista de comandos perigosos.
- [ ] Automatizar execução de suítes de teste a partir de gatilhos configuráveis (ex.: após comandos `pip install`).
- [ ] Criar modo TUI/observabilidade para acompanhar iterações em tempo real.
- [ ] Auto-propor benchmark tasks por capability e validar no pipeline.

## Notas
- Precisaremos da API key OpenRouter (via env, ex.: `OPENROUTER_API_KEY`). Avisar o usuário antes da integração final.
- Confirmação do(s) modelo(s) a serem usados (ex.: `anthropic/claude-3.5-sonnet`, `openai/gpt-4o-mini`) ajuda a validar payloads.
- Checar requisitos de dependências adicionais (`httpx`, `pydantic?`) antes de atualizar `pyproject.toml`.
