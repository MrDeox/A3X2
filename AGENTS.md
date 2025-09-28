# Repository Guidelines

## Project Structure & Module Organization
- `a3x/`: núcleo do agente, com CLI, ciclo autônomo (`autoloop.py`), planners, executores e memória semântica (`memory/`).
- `configs/`: presets YAML para runs (`sample.yaml`), pipelines de seeds e scripts manuais; derive sempre a partir deles.
- `seed/`: backlog, métricas, relatos de execução e memória longa (`seed/memory/`). Artefatos derivados ficam em `seed/evaluations/`.
- `tests/`: suíte pytest espelha `a3x/`; arquivos em `tests/generated/` são gerados automaticamente.
- `docs/`: notas de arquitetura e roadmap; cite-os em issues em vez de duplicar conteúdo.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: prepara ambiente isolado.
- `pip install -e .[dev]`: instala runtime, pytest, ruff e black.
- `a3x run --goal "Adicionar endpoint /health" --config configs/sample.yaml`: executa ciclo completo com preset padrão.
- `a3x seed run --config configs/seed_manual.yaml`: drena a próxima seed pendente do backlog.
- `a3x autopilot --cycles 3 --goals seed/goal_rotation.yaml`: alterna objetivos e monitora seeds automaticamente.
- `pytest -q`, `ruff check a3x tests`, `black a3x tests`: valide antes de abrir PR.

## Coding Style & Naming Conventions
- Python 3.10+, indentação de 4 espaços; novas APIs públicas com type hints.
- Funções/módulos em `snake_case`, classes em `PascalCase`, configs serializadas em kebab-case.
- Prefira `dataclasses` ou `TypedDict` para payloads estruturados; mensagens CLI permanecem em português.
- Confie no `black`; use `# noqa` apenas com justificativa curta.

## Testing Guidelines
- Nomeie arquivos como `tests/test_<modulo>.py` e reutilize fixtures existentes.
- Inclua cenários negativos para planners, executores e memória ao introduzir capacidades.
- Rode `pytest -q` antes de commitar e mantenha `tests/generated/` sob controle automático.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `refactor:`) com título objetivo em português.
- PRs devem listar arquivos/configs tocados, resultados (`pytest`, `a3x run`, `a3x autopilot`) e seeds impactadas.
- Anexe logs relevantes e aponte tickets ou docs (`docs/roadmap_agi.md`) quando aplicável.

## Security & Configuration Tips
- Carregue `OPENROUTER_API_KEY` via `.env` ignorado; não versione segredos.
- Revise `configs/*.yaml` antes de habilitar comandos com efeitos colaterais; mantenha timeouts e operações idempotentes.
- Redija entradas de memória (`seed/memory/`) sem dados sensíveis e monitore tamanho para evitar deriva.
