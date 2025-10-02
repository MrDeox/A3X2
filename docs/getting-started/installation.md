# 📦 Installation Guide

This guide provides detailed installation instructions for different use cases and environments.

## System Requirements

- **Operating System**: Linux, macOS, or Windows (WSL recommended for Windows)
- **Python**: 3.10 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Internet connection for AI model access

## Quick Installation

If you're in a hurry, see our [Quick Start Guide](quick-start.md) for the fastest setup.

## Detailed Installation

### Step 1: Get the Code

**Option A: Clone from Git**
```bash
git clone <repository-url> a3x-project
cd a3x-project
```

**Option B: Download ZIP**
1. Download the repository as a ZIP file
2. Extract to your desired location
3. Open terminal and navigate to the extracted folder

### Step 2: Set Up Python Environment

**Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Verify Python version:**
```bash
python --version  # Should show Python 3.10+
```

### Step 3: Install Dependencies

**Basic installation:**
```bash
pip install -e .
```

**Development installation (includes testing tools):**
```bash
pip install -e .[dev]
```

**Full installation (includes all optional dependencies):**
```bash
pip install -e .[all]
```

### Step 4: Verify Installation

```bash
a3x --help
```

You should see the available commands and options.

## Configuration

### API Key Setup

A3X requires an API key for AI model access:

1. **Get an API key** from [OpenRouter.ai](https://openrouter.ai/)
2. **Set environment variable:**
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

3. **Or create `.env` file:**
   ```bash
   echo "OPENROUTER_API_KEY=your-key-here" > .env
   ```

### Configuration Files

A3X uses YAML configuration files in the `configs/` directory:

- **`configs/sample.yaml`**: Ready-to-use configuration with OpenRouter
- **`configs/manual.yaml`**: Deterministic testing without external APIs
- **`configs/seed_manual.yaml`**: Configuration for autonomous seed execution

## Optional: IDE Setup

### VS Code

1. **Install Python extension**
2. **Select interpreter**: Choose your virtual environment
3. **Install recommended extensions**:
   - Python (Microsoft)
   - Pylint
   - YAML

### PyCharm / IntelliJ

1. **Open project folder**
2. **Set Python interpreter** to your virtual environment
3. **Enable linting** for YAML files

## Troubleshooting Installation

### Common Issues

**"Python 3.10+ required"**
```bash
# Check if you have multiple Python versions
ls /usr/bin/python*
# Use specific version
/usr/bin/python3.11 -m venv .venv
```

**"Module not found" errors**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate
# Reinstall
pip install -e . --force-reinstall
```

**"No module named 'a3x'" after installation**
```bash
# Check if you're in the right directory
pwd  # Should be the project root
# Verify setup.py exists
ls setup.py
```

**Permission errors on Linux/macOS**
```bash
# Install in user space
pip install -e . --user
```

### Getting Help

- Check the [Troubleshooting Guide](../troubleshooting/common-issues.md)
- Review error messages carefully
- Ensure all prerequisites are met
- Try the minimal installation first

## Next Steps

1. ✅ Complete installation
2. 🔄 Set up API key
3. 🚀 Run your [first task](../getting-started/first-run.md)
4. 📚 Explore [examples and tutorials](../user-guide/examples/)

## Advanced Installation Options

### Docker Installation (Coming Soon)

For containerized deployments, Docker support is planned for future releases.

### Offline Installation

For air-gapped environments, A3X can be configured to work with local models through Ollama or similar services.

### Custom Model Providers

A3X supports multiple AI model providers:

- **OpenRouter**: Default, supports many models
- **OpenAI**: Direct OpenAI API access
- **Local models**: Ollama, LM Studio, etc.
- **Manual mode**: Pre-defined scripts for testing

See the [Configuration Guide](../user-guide/configuration.md) for provider-specific setup instructions.