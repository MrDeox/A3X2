# A3X Configuration System

A comprehensive, validated, and maintainable configuration system for the A3X autonomous agent with built-in migration support and testing utilities.

## Overview

The configuration system provides:
- **JSON Schema validation** with detailed error reporting
- **Configuration migration** for backward compatibility
- **Comprehensive testing** utilities for validation and functional testing
- **Configuration analysis** tools for troubleshooting and optimization
- **Template generation** for different use cases
- **CLI integration** for easy configuration management

## Architecture

```
a3x/config/
├── validation/           # Core validation framework
│   ├── __init__.py      # Public API
│   ├── schemas/         # JSON Schema definitions
│   ├── validator.py     # Validation engine
│   └── migration.py     # Migration tools
├── testing/             # Configuration testing utilities
│   ├── __init__.py
│   └── config_tester.py # Testing framework
├── utils/               # Analysis and generation tools
│   ├── __init__.py
│   ├── analyzer.py      # Configuration analysis
│   └── generator.py     # Template generation
└── README.md           # This documentation
```

## Quick Start

### 1. Basic Configuration Validation

```bash
# Validate a configuration file
a3x config validate configs/sample.yaml

# Test configuration comprehensively
a3x config test configs/sample.yaml

# Analyze configuration for issues and optimizations
a3x config analyze configs/sample.yaml
```

### 2. Configuration Migration

```bash
# Migrate configuration to latest version
a3x config migrate configs/old_config.yaml

# Migrate to specific version with backup
a3x config migrate --target-version 1.2.0 --no-backup configs/config.yaml
```

### 3. Generate Configuration Templates

```bash
# Generate minimal configuration
a3x config generate minimal my_config.yaml

# Generate development-ready configuration
a3x config generate development dev_config.yaml

# Generate security-hardened configuration
a3x config generate secure secure_config.yaml

# Generate LLM provider examples
a3x config generate llm-examples ./llm_configs/

# Generate schema documentation
a3x config generate schema-docs CONFIG_SCHEMA.md
```

## Configuration Sections

### LLM Configuration

```yaml
llm:
  type: "openrouter"  # openai, anthropic, openrouter, ollama, manual
  model: "x-ai/grok-4-fast:free"
  endpoint: "https://openrouter.ai/api/v1"
  api_key_env: "OPENROUTER_API_KEY"
  base_url: "https://openrouter.ai/api/v1"
```

**Validation Rules:**
- `type` is required and must be one of: `openai`, `anthropic`, `openrouter`, `ollama`, `manual`
- `model` is required for non-manual types
- `api_key_env` is required for API-based providers
- URLs must be valid format

### Workspace Configuration

```yaml
workspace:
  root: "."                    # Workspace root directory
  allow_outside_root: false   # Allow operations outside workspace
```

**Validation Rules:**
- `root` path must exist (warning if not)
- `allow_outside_root` must be boolean

### Limits Configuration

```yaml
limits:
  max_iterations: 50     # 1-1000, default: 50
  command_timeout: 120   # 1-3600 seconds, default: 120
  max_failures: 10       # 1-100, default: 10
  total_timeout: 3600    # Optional, 1-86400 seconds
```

**Validation Rules:**
- All numeric values must be positive integers
- Values must be within specified ranges

### Policies Configuration

```yaml
policies:
  allow_network: false         # Allow network access
  allow_shell_write: true     # Allow shell write operations
  deny_commands:              # Commands to deny
    - "rm -rf"
    - "sudo"
    - "su"
```

**Validation Rules:**
- All fields must be boolean or string arrays
- Dangerous commands should be in deny list

### Goals Configuration

```yaml
goals:
  apply_patch_success_rate: 0.9     # Simple threshold format
  actions_success_rate:
    min: 0.85                      # Object format with min threshold
  tests_success_rate:
    min: 0.95
```

**Validation Rules:**
- Thresholds must be numbers between 0.0 and 1.0
- Both simple and object formats supported

## Advanced Usage

### Configuration Testing

The testing framework provides multiple test types:

```bash
# Run all tests
a3x config test my_config.yaml

# Run only validation tests
a3x config test --validation-only my_config.yaml

# Run only functional tests
a3x config test --functional my_config.yaml
```

**Test Categories:**
1. **Schema Validation**: JSON Schema compliance
2. **File Existence**: Check referenced files exist
3. **Configuration Loading**: Can configuration be loaded
4. **Value Ranges**: Values within acceptable ranges
5. **Dependencies**: Required dependencies satisfied
6. **Command Execution**: Test commands are available
7. **Environment Variables**: Required env vars are set

### Configuration Analysis

Get detailed insights about your configuration:

```bash
# Analyze and print report
a3x config analyze my_config.yaml

# Export analysis to JSON
a3x config analyze my_config.yaml --output analysis.json
```

**Analysis Categories:**
- **Security**: Overly permissive settings, missing deny rules
- **Performance**: High limits that may impact performance
- **Maintainability**: Absolute paths, missing sections
- **Best Practices**: Unrealistic thresholds, misconfigurations
- **Dependencies**: Missing required fields

### Configuration Migration

Safely upgrade configurations between versions:

```bash
# Migrate to latest version (creates backup)
a3x config migrate old_config.yaml

# Migrate to specific version without backup
a3x config migrate --target-version 1.2.0 --no-backup config.yaml

# Check what migrations would be applied
a3x config analyze old_config.yaml  # Shows version detection
```

**Migration Features:**
- Automatic version detection
- Safe migration with backups
- Detailed migration logging
- Rollback support via backups

## Schema Documentation

### Complete Schema Reference

The configuration schema supports the following sections:

#### LLM Configuration (`llm`)
- **type** (required): LLM provider type
- **model**: Model name or identifier
- **script**: Path to custom script (manual type only)
- **endpoint**: API endpoint URL
- **api_key_env**: Environment variable for API key
- **base_url**: Base URL for API requests

#### Workspace Configuration (`workspace`)
- **root**: Workspace root directory path
- **allow_outside_root**: Allow operations outside workspace

#### Limits Configuration (`limits`)
- **max_iterations**: Maximum iterations (1-1000)
- **command_timeout**: Command timeout in seconds (1-3600)
- **max_failures**: Maximum failures (1-100)
- **total_timeout**: Total timeout in seconds (optional)

#### Tests Configuration (`tests`)
- **auto**: Enable automatic test execution
- **commands**: Test commands (string or array format)

#### Policies Configuration (`policies`)
- **allow_network**: Allow network access
- **allow_shell_write**: Allow shell write operations
- **deny_commands**: Commands to deny execution

#### Goals Configuration (`goals`)
- Pattern: `goal_name: threshold` or `goal_name: {min: threshold}`
- Thresholds: Numbers between 0.0 and 1.0

#### Loop Configuration (`loop`)
- **auto_seed**: Enable automatic seed generation
- **seed_backlog**: Seed backlog file path
- **seed_config**: Seed configuration file path
- **seed_interval**: Interval between seeds (seconds)
- **seed_max_runs**: Maximum seed runs
- **stop_when_idle**: Stop when idle
- **use_memory**: Enable memory usage
- **memory_top_k**: Number of memories to retrieve
- **interactive**: Enable interactive mode

#### Audit Configuration (`audit`)
- **enable_file_log**: Enable file logging
- **file_dir**: Audit files directory
- **enable_git_commit**: Enable git commits
- **commit_prefix**: Git commit prefix

#### Scaling Configuration (`scaling`)
- **cpu_threshold**: CPU threshold for scaling (0.0-1.0)
- **memory_threshold**: Memory threshold for scaling (0.0-1.0)
- **max_recursion_adjust**: Maximum recursion adjustments

## Error Handling

### Validation Errors

When validation fails, you'll see detailed error messages:

```
❌ Configuration validation failed: configs/invalid.yaml
Configuration file: configs/invalid.yaml

Validation errors:
  1. llm -> type: 'invalid_type' is not one of ['openai', 'anthropic', 'openrouter', 'ollama', 'manual']
  2. limits -> max_iterations: 1500 is greater than the maximum of 1000
  3. workspace -> root: /nonexistent/path does not exist
```

### Common Issues and Solutions

#### Issue: "JSON schema validation not available"
**Solution**: Install the required dependency:
```bash
pip install jsonschema
```

#### Issue: Configuration file not found
**Solution**: Check file path and ensure it exists:
```bash
ls -la configs/my_config.yaml
a3x config validate configs/my_config.yaml
```

#### Issue: Environment variable not set
**Solution**: Set required environment variables:
```bash
export OPENROUTER_API_KEY="your-api-key"
a3x config test my_config.yaml
```

## Examples

### Example: Production Configuration

```yaml
llm:
  type: "openrouter"
  model: "x-ai/grok-4-fast:free"
  base_url: "https://openrouter.ai/api/v1"
  api_key_env: "OPENROUTER_API_KEY"

workspace:
  root: "/home/user/projects/myproject"
  allow_outside_root: false

limits:
  max_iterations: 100
  command_timeout: 300
  max_failures: 5
  total_timeout: 7200

policies:
  allow_network: true
  allow_shell_write: false
  deny_commands: ["rm -rf", "sudo", "su", "chmod 777"]

goals:
  apply_patch_success_rate: {min: 0.95}
  actions_success_rate: {min: 0.9}
  tests_success_rate: {min: 0.98}

audit:
  enable_file_log: true
  file_dir: "seed/changes"
  enable_git_commit: true
  commit_prefix: "A3X"
```

### Example: Development Configuration

```yaml
llm:
  type: "manual"
  script: "echo 'Development mode: manual execution'"

workspace:
  root: "."
  allow_outside_root: true

limits:
  max_iterations: 25
  command_timeout: 120
  max_failures: 10

tests:
  auto: true
  commands:
    - ["pytest", "tests/", "-v"]
    - ["ruff", "check", "."]
    - ["black", "--check", "."]

policies:
  allow_network: true
  allow_shell_write: true
  deny_commands: ["rm -rf"]

goals:
  tests_success_rate: {min: 0.9}
```

## CLI Reference

### `a3x config validate`
Validate a configuration file against the schema.

```bash
a3x config validate CONFIG_FILE [--lenient]
```

### `a3x config test`
Test a configuration file comprehensively.

```bash
a3x config test CONFIG_FILE [--functional | --validation-only]
```

### `a3x config analyze`
Analyze configuration for issues and optimizations.

```bash
a3x config analyze CONFIG_FILE [--output FILE]
```

### `a3x config migrate`
Migrate configuration to a newer version.

```bash
a3x config migrate CONFIG_FILE [--target-version VERSION] [--no-backup]
```

### `a3x config generate`
Generate configuration templates.

```bash
a3x config generate TYPE OUTPUT
```

**Types:**
- `minimal`: Minimal configuration
- `development`: Development-ready configuration
- `secure`: Security-hardened configuration
- `llm-examples`: Example configurations for different LLM providers
- `schema-docs`: Schema documentation

## Integration with A3X

The configuration validation system is automatically integrated into the main A3X workflow:

1. **Automatic Validation**: Configurations are validated when loaded
2. **Detailed Errors**: Clear error messages help identify issues
3. **Migration Support**: Old configurations are automatically migrated
4. **Development Tools**: Rich tooling for configuration management

## Best Practices

### 1. Use Relative Paths
```yaml
# ✅ Good
workspace:
  root: "."
loop:
  seed_backlog: "seed/backlog.yaml"

# ❌ Avoid
workspace:
  root: "/absolute/path/to/project"
```

### 2. Set Realistic Goals
```yaml
# ✅ Good
goals:
  apply_patch_success_rate: {min: 0.9}
  tests_success_rate: {min: 0.95}

# ❌ Avoid (too strict)
goals:
  apply_patch_success_rate: {min: 0.99}
```

### 3. Secure by Default
```yaml
# ✅ Good
policies:
  allow_network: false
  allow_shell_write: false
  deny_commands: ["rm -rf", "sudo", "su"]
```

### 4. Use Appropriate Limits
```yaml
# ✅ Good for development
limits:
  max_iterations: 50
  command_timeout: 120

# ✅ Good for production
limits:
  max_iterations: 100
  command_timeout: 300
  total_timeout: 7200
```

## Troubleshooting

### Common Issues

1. **"Configuration validation failed"**
   - Check the error details for specific field issues
   - Use `a3x config analyze` for comprehensive analysis
   - Verify all required fields are present

2. **"File does not exist"**
   - Check file paths in configuration
   - Use relative paths when possible
   - Ensure referenced files exist

3. **"Environment variable not set"**
   - Set required API keys as environment variables
   - Use `a3x config test` to identify missing variables

4. **"Migration failed"**
   - Check backup was created before migration
   - Review migration messages for specific issues
   - Consider manual fixes if automatic migration fails

### Getting Help

1. **Analyze your configuration**: `a3x config analyze config.yaml`
2. **Test thoroughly**: `a3x config test config.yaml`
3. **Check examples**: `a3x config generate development example.yaml`
4. **Review schema docs**: `a3x config generate schema-docs SCHEMA.md`

## Contributing

When adding new configuration options:

1. Update the JSON schema in `a3x/config/validation/schemas/`
2. Add validation rules in `a3x/config/validation/validator.py`
3. Update migration logic in `a3x/config/validation/migration.py`
4. Add tests in `tests/unit/a3x/test_config*.py`
5. Update this documentation

## Version History

- **v1.0.0**: Initial configuration system
- **v1.1.0**: Added scaling and audit sections
- **v1.2.0**: Restructured goals format
- **v1.3.0**: Enhanced security policies and validation

The configuration system automatically handles migrations between versions.