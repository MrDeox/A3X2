# Repository Guidelines

## Project Structure & Module Organization
- `a3x/` centraliza loop, CLI, executores e clientes; agrupe novas features em submódulos claros.
- `configs/` mantém presets YAML; derive deles ao registrar novos modos ou provedores.
- `docs/` concentra notas de arquitetura e manifestos; cite em issues em vez de duplicar conteúdo.
- `seed/` armazena backlog, métricas e relatórios do ciclo autônomo; mantenha artefatos derivados em `seed/evaluations/`.
- `tests/` espelha `a3x/`; adicione casos manuais ao lado dos módulos e não edite `tests/generated/` manualmente.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` prepara o ambiente isolado.
- `pip install -e .[dev]` instala runtime e toolchain (`pytest`, `ruff`, `black`).
- `a3x run --goal "Adicionar endpoint /health" --config configs/sample.yaml` executa um ciclo completo com o preset padrão.
- `a3x seed run --config configs/seed_manual.yaml` processa seeds pendentes e atualiza relatórios.
- `pytest` roda a suíte; use `pytest tests/generated -k nome` ao iterar sobre testes adaptativos.
- `ruff check a3x tests` e `black a3x tests` fixam lint/format antes de abrir PR.

## Coding Style & Naming Conventions
- Código em Python 3.10+, indentação de 4 espaços e type hints para novas APIs públicas.
- Funções e módulos em `snake_case`, classes em `PascalCase`, arquivos de config seguem `kebab-case` apenas quando serializados.
- Prefira `dataclasses` ou `TypedDicts` para payloads estruturados e mantenha mensagens CLI voltadas ao usuário final em português.
- Confie no `black`; use `# noqa` somente com comentário curto justificando o desvio.

## Testing Guidelines
- Estruture arquivos como `test_<modulo>.py` e reutilize fixtures existentes para cobrir caminhos críticos.
- Para novas capacidades, adicione testes manuais em `tests/` e deixe `tests/generated/` sob controle dos geradores.
- Antes de commitar, execute `pytest -q`; inclua cenários negativos que validem decisões do agente ou safe-guards do executor.

## Commit & Pull Request Guidelines
- Siga Conventional Commits (`feat:`, `fix:`, `refactor:`) com título objetivo; detalhe contexto extra em português no corpo.
- Liste arquivos ou configs tocados (`configs/sample.yaml`, `seed/backlog.yaml`) e resultados de testes (`pytest`, `a3x run`) na descrição.
- PRs devem apontar tickets relevantes, anexar logs que evidenciem mudanças de comportamento e mencionar impactos em seeds ou métricas.

## Security & Configuration Tips
- Não versionar segredos; carregue `OPENROUTER_API_KEY` via variáveis de ambiente ou `.env` ignorado.
- Revise `configs/*.yaml` antes de habilitar comandos com efeitos colaterais; novos executores devem preservar timeouts e operações idempotentes.
