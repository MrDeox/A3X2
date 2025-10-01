# A3X Configuration Files

The `configs/` directory contains YAML presets that define the behavior of the A3X agent for different execution modes. These configurations are essential for customizing LLM integration, workspace constraints, policy enforcement, goal thresholds, and loop parameters. All configs are derived from a common schema to ensure consistency across runs.

Presets are loaded via CLI flags (e.g., `--config configs/sample.yaml`) and support overrides for dynamic adaptation. The directory structure includes:

- **Root configs**: General-purpose files like `sample.yaml` for standard agent runs.
- **Seed configs** (`configs/seed_*.yaml`): Specialized for seed processing, curriculum-based learning, and autonomous evolution (e.g., `seed_manual.yaml` for manual backlog drainage, `seed_testing_curriculum.yaml` for test-focused iterations).
- **Script configs** (`configs/scripts/*.yaml`): Targeted presets for demos, linting, or specific improvements (e.g., `run_pytest.yaml` for validation runs).
- **Supporting files**: `README.md` (this file), example schemas, and JSON hints for policy tuning.

Configs are versioned and should be reviewed before enabling side-effect commands (e.g., git commits). Use `a3x run --config <path>` to execute with a preset.

## YAML Schema

Configs follow a structured YAML schema with top-level sections. Required fields are marked; optional ones have defaults. Use type hints for validation (e.g., via Pydantic in code).

### Core Sections

- **llm** (required): LLM client configuration.
  - `type` (str): Client type (e.g., "openrouter", "manual").
  - `model` (str): Model identifier (e.g., "x-ai/grok-4-fast:free").
  - `base_url` (str, optional): API endpoint (default: provider-specific).
  - `api_key_env` (str, optional): Environment variable for API key (e.g., "OPENROUTER_API_KEY").
  
  Example:
  ```
  llm:
    type: openrouter
    model: "x-ai/grok-4-fast:free"
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
  ```

- **workspace** (required): Workspace settings.
  - `root` (str): Root directory (relative or absolute, default: ".").
  - `allow_outside_root` (bool): Permit operations outside root (default: false for security).

  Example:
  ```
  workspace:
    root: .
    allow_outside_root: true
  ```

- **limits** (required): Execution bounds to prevent runaway runs.
  - `max_iterations` (int): Maximum loop iterations (default: 25).
  - `command_timeout` (int, seconds): Shell command timeout (default: 120).
  - `max_failures` (int): Maximum consecutive failures before abort (default: 5).

  Example (from `sample.yaml`):
  ```
  limits:
    max_iterations: 25
    command_timeout: 120
    max_failures: 5
  ```

- **tests** (optional): Automated testing configuration.
  - `auto` (bool): Run tests after patches/writes (default: false).
  - `commands` (list[str]): Commands to execute (e.g., ["pytest -q"]).

  Example:
  ```
  tests:
    auto: false
    commands: ["pytest -q", "ruff check ."]
  ```

- **policies** (required): Security and behavior policies.
  - `allow_network` (bool): Permit network access (default: false).
  - `allow_shell_write` (bool): Allow file writes via shell (default: true).
  - `deny_commands` (list[str], optional): Patterns to block (e.g., ["rm -rf"]).

  Example:
  ```
  policies:
    allow_network: true
    allow_shell_write: true
  ```

- **goals** (optional): Metric thresholds for evaluation.
  - Keys are metric names (e.g., `apply_patch_success_rate`); values are dicts with `min` (float).

  Example (from `sample.yaml`):
  ```
  goals:
    apply_patch_success_rate:
      min: 0.9
    actions_success_rate:
      min: 0.85
    tests_success_rate:
      min: 0.95
  ```

- **loop** (optional): Autonomous loop parameters.
  - `auto_seed` (bool): Enable automatic seed processing (default: true).
  - `seed_backlog` (str): Path to backlog YAML (default: "../seed/backlog.yaml").
  - `seed_config` (str): Seed preset (e.g., "seed_manual.yaml").
  - `seed_interval` (int): Seconds between seed checks (default: 0 for continuous).
  - `stop_when_idle` (bool): Halt on empty backlog (default: true).
  - `use_memory` (bool): Enable semantic memory (default: false).
  - `memory_top_k` (int): Top memories to retrieve (default: 3).

  Example:
  ```
  loop:
    auto_seed: true
    seed_backlog: ../seed/backlog.yaml
    seed_config: seed_manual.yaml
    seed_interval: 0
    stop_when_idle: true
    use_memory: false
    memory_top_k: 3
  ```

- **audit** (optional): Logging and versioning.
  - `enable_file_log` (bool): Log changes to files (default: true).
  - `file_dir` (str): Log directory (default: "seed/changes").
  - `enable_git_commit` (bool): Auto-commit changes (default: true).
  - `commit_prefix` (str): Git commit prefix (default: "A3X").

  Example (from `sample.yaml`):
  ```
  audit:
    enable_file_log: true
    file_dir: seed/changes
    enable_git_commit: true
    commit_prefix: "A3X"
  ```

Validation: Load configs via `AgentConfig.from_yaml(path)` in code; it enforces types and defaults. Errors raise `ConfigError` with details.

## Examples

### Basic Run (sample.yaml)
This preset configures a standard agent loop with OpenRouter LLM, moderate limits, and basic goals. Suitable for general development tasks.

Full content:
```
llm:
  type: openrouter
  model: "x-ai/grok-4-fast:free"
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY

workspace:
  root: .
  allow_outside_root: true

limits:
  max_iterations: 25
  command_timeout: 120
  max_failures: 5

tests:
  auto: false
  commands: []

policies:
  allow_network: true
  allow_shell_write: true

goals:
  apply_patch_success_rate:
    min: 0.9
  actions_success_rate:
    min: 0.85
  tests_success_rate:
    min: 0.95

loop:
  auto_seed: true
  seed_backlog: ../seed/backlog.yaml
  seed_config: seed_manual.yaml
  seed_interval: 0
  stop_when_idle: true
  use_memory: false
  memory_top_k: 3

audit:
  enable_file_log: true
  file_dir: seed/changes
  enable_git_commit: true
  commit_prefix: "A3X"
```

Usage: `a3x run --goal "Implement feature X" --config configs/sample.yaml`

### Seed Curriculum (seed_testing_curriculum.yaml)
For test-focused evolution: High test thresholds, auto-testing enabled, and seed interval for curriculum steps.

Key overrides:
```
tests:
  auto: true
  commands: ["pytest -q --cov", "ruff check ."]

goals:
  tests_success_rate:
    min: 0.98  # Strict for curriculum

loop:
  seed_config: seed_testing_curriculum_step1.yaml  # Chain steps
  seed_interval: 300  # 5min between evolutions
```

Usage: `a3x seed run --config configs/seed_testing_curriculum.yaml`

### Script Preset (scripts/run_lint.yaml)
Minimal config for linting runs: Short timeouts, no network, focus on code quality metrics.

```
limits:
  max_iterations: 5
  command_timeout: 30

policies:
  allow_network: false

goals:
  lint_success_rate:
    min: 1.0
```

Usage: `a3x run --goal "Run linting" --config configs/scripts/run_lint.yaml`

## Usage Patterns for Presets

### 1. Standard Development
- Load `sample.yaml` as base.
- Override goals for project-specific thresholds (e.g., higher patch success).
- Enable auto-tests for CI-like validation.
- Command: `a3x run --config configs/sample.yaml --goal "Refactor module Y"`

### 2. Seed-Driven Autonomy
- Use seed presets for curriculum or manual backlog.
- Set `loop.auto_seed: true` and point to backlog.
- Monitor via `seed/reports/`; chain configs (e.g., step1 → step2).
- Command: `a3x autopilot --cycles 3 --config configs/seed_testing_curriculum.yaml`

### 3. Targeted Scripts
- Create lightweight YAML in `scripts/` for one-off tasks (e.g., docs generation).
- Disable audits if non-persistent.
- Validate: `a3x plan --dry-run --config configs/scripts/patch_doc.yaml`
- Run: `a3x run --config configs/scripts/improve_calculator.yaml`

### Best Practices
- **Security**: Set `allow_network: false` unless needed; review `deny_commands`.
- **Customization**: Derive new presets from `sample.yaml`; use kebab-case keys.
- **Validation**: Run `pytest tests/unit/a3x/test_config.py` after changes.
- **Timeouts & Limits**: Adjust based on hardware; monitor costs via LLM metrics.
- **Integration**: Load in code: `config = AgentConfig.from_yaml("configs/sample.yaml")`.

For advanced tuning, see `docs/api.md` (AgentConfig) and `seed/policy_hints.json`. Report issues in `seed/backlog.yaml`.