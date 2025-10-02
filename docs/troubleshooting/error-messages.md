# 📋 Error Message Reference

This guide provides detailed explanations for common error messages and their solutions.

## Authentication & API Errors

### "ERROR: Invalid API key provided"

**Meaning**: The OpenRouter API key is missing, invalid, or expired.

**Context**: Usually appears during A3X initialization.

**Solutions**:
1. **Check API key exists**:
   ```bash
   echo $OPENROUTER_API_KEY | head -c 20  # Should show key prefix
   ```

2. **Verify key in OpenRouter dashboard**:
   - Go to [OpenRouter.ai/keys](https://openrouter.ai/keys)
   - Ensure key is active and has credits

3. **Generate new key**:
   - Create new key in dashboard
   - Update environment variable or `.env` file

### "ERROR: Model 'x-ai/grok-4-fast:free' not found"

**Meaning**: The specified AI model is not available or doesn't exist.

**Context**: Appears when loading configuration.

**Solutions**:
1. **Check available models**:
   ```bash
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        https://openrouter.ai/api/v1/models | jq '.data[].id'
   ```

2. **Update config file**:
   ```yaml
   # In configs/sample.yaml
   llm:
     model: "x-ai/grok-beta"  # Try different model
   ```

3. **Use fallback model**:
   ```bash
   a3x run --goal "test" --config configs/manual.yaml
   ```

### "ERROR: Rate limit exceeded"

**Meaning**: Too many API requests sent too quickly.

**Context**: Usually during intensive operations.

**Solutions**:
1. **Wait and retry**:
   - OpenRouter typically resets limits every minute/hour

2. **Use slower model**:
   ```yaml
   llm:
     model: "openai/gpt-3.5-turbo"  # Often higher rate limits
   ```

3. **Add delays in config**:
   ```yaml
   execution:
     retry_delay: 5  # Seconds between retries
   ```

## Configuration Errors

### "ERROR: Configuration file not found: configs/sample.yaml"

**Meaning**: The specified config file doesn't exist or path is wrong.

**Context**: Appears when starting A3X with `--config` flag.

**Solutions**:
1. **Check file exists**:
   ```bash
   ls -la configs/
   ```

2. **Use absolute path**:
   ```bash
   a3x run --goal "test" --config $(pwd)/configs/sample.yaml
   ```

3. **Check current directory**:
   ```bash
   pwd  # Should be project root
   ```

### "ERROR: YAML parsing failed in configs/sample.yaml"

**Meaning**: The YAML configuration file has syntax errors.

**Context**: Appears during config loading.

**Solutions**:
1. **Validate YAML syntax**:
   ```bash
   python -c "import yaml; print(yaml.safe_load(open('configs/sample.yaml')))"
   ```

2. **Check for common issues**:
   - Indentation (use spaces, not tabs)
   - Missing quotes around special characters
   - Incorrect list syntax

3. **Use online validator**:
   - Copy content to [YAML Parser](https://yaml-online-parser.appspot.com/)

### "ERROR: Invalid configuration schema"

**Meaning**: Configuration values don't match expected format.

**Context**: Appears after YAML parsing.

**Solutions**:
1. **Check data types**:
   ```yaml
   # Wrong
   max_iterations: "10"  # String instead of number

   # Correct
   max_iterations: 10    # Number
   ```

2. **Validate against schema**:
   - Compare with working config files

3. **Start with minimal config**:
   ```bash
   a3x run --goal "test" --config configs/manual.yaml
   ```

## Execution Errors

### "ERROR: Command execution failed: timeout"

**Meaning**: A system command took too long to complete.

**Context**: Appears during code execution or testing.

**Solutions**:
1. **Increase timeout**:
   ```yaml
   execution:
     command_timeout: 300  # Increase from default (60)
   ```

2. **Check command manually**:
   ```bash
   cd /path/to/project
   timeout 60s your-command  # Test with timeout
   ```

3. **Optimize command**:
   - Break long-running commands into smaller parts

### "ERROR: File permission denied"

**Meaning**: A3X can't read/write files due to permissions.

**Context**: Appears during file operations.

**Solutions**:
1. **Check file permissions**:
   ```bash
   ls -la problematic-file
   chmod 644 problematic-file
   ```

2. **Check directory permissions**:
   ```bash
   ls -ld /path/to/directory
   chmod 755 /path/to/directory
   ```

3. **Run as appropriate user**:
   - Ensure A3X runs with needed permissions

### "ERROR: Disk space exhausted"

**Meaning**: Not enough disk space for operations.

**Context**: Appears during large file operations or logging.

**Solutions**:
1. **Check disk usage**:
   ```bash
   df -h
   du -sh * | sort -hr | head -10
   ```

2. **Clean up space**:
   ```bash
   # Remove temporary files
   rm -rf __pycache__/ .pytest_cache/ *.pyc
   # Clear large logs if safe
   truncate -s 0 large-log-file.log
   ```

3. **Use external storage**:
   - Move large files to external drive temporarily

## Code Generation Errors

### "ERROR: Syntax error in generated code"

**Meaning**: The AI generated code with syntax errors.

**Context**: Appears when trying to execute generated code.

**Solutions**:
1. **Review generated code**:
   - Look for obvious syntax issues

2. **Ask for correction**:
   ```bash
   a3x run --goal "Fix syntax errors in the generated code" --config configs/sample.yaml
   ```

3. **Use different model**:
   ```yaml
   llm:
     model: "anthropic/claude-3-sonnet"  # Often better at syntax
   ```

### "ERROR: Import resolution failed"

**Meaning**: Generated code references non-existent modules.

**Context**: Appears when importing dependencies.

**Solutions**:
1. **Install missing packages**:
   ```bash
   pip install missing-package-name
   ```

2. **Check import paths**:
   - Verify module names are spelled correctly
   - Check if modules are installed

3. **Use virtual environment**:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Runtime & Performance Errors

### "ERROR: Out of memory"

**Meaning**: System ran out of available memory.

**Context**: Appears during large operations or with big codebases.

**Solutions**:
1. **Monitor memory usage**:
   ```bash
   free -h  # Linux
   top     # Real-time monitoring
   ```

2. **Reduce memory usage**:
   ```yaml
   # In config
   history:
     max_context_length: 2000  # Reduce context size
   ```

3. **Process in chunks**:
   - Break large tasks into smaller pieces

### "ERROR: Process terminated"

**Meaning**: A3X process was killed or crashed.

**Context**: Can appear during long-running operations.

**Solutions**:
1. **Check system logs**:
   ```bash
   dmesg | tail -20  # Linux system logs
   journalctl -u a3x  # If running as service
   ```

2. **Look for crash dumps**:
   ```bash
   find . -name "*.dump" -o -name "core" 2>/dev/null
   ```

3. **Restart with debug mode**:
   ```bash
   a3x run --goal "test" --config configs/sample.yaml --verbose
   ```

## Network & Connectivity Errors

### "ERROR: Network unreachable"

**Meaning**: Can't connect to required services.

**Context**: Appears when API calls fail.

**Solutions**:
1. **Test connectivity**:
   ```bash
   ping 8.8.8.8
   curl -I https://openrouter.ai
   ```

2. **Check proxy settings**:
   ```bash
   env | grep -i proxy
   ```

3. **Use local models**:
   ```yaml
   llm:
     type: "ollama"  # If Ollama is running locally
     model: "codellama"
   ```

### "ERROR: SSL certificate verification failed"

**Meaning**: Can't verify SSL certificates for HTTPS connections.

**Context**: Appears with corporate networks or VPNs.

**Solutions**:
1. **Check system certificates**:
   ```bash
   curl -v https://openrouter.ai/api/v1/models
   ```

2. **Disable SSL verification** (not recommended):
   ```bash
   export CURL_SSL_NO_VERIFY=1
   ```

3. **Update certificates**:
   ```bash
   sudo apt-get update-ca-certificates  # Ubuntu/Debian
   ```

## Getting More Help

### Debug Mode

Run A3X with verbose output to get more details:

```bash
a3x run --goal "test" --config configs/sample.yaml --verbose --debug
```

### Log Files

Check for log files that might contain more details:

```bash
find . -name "*.log" -exec tail -20 {} \; 2>/dev/null
```

### Reporting Issues

When reporting errors, please include:

1. **Complete error message**
2. **Command that triggered the error**
3. **Configuration file** (sanitized)
4. **Environment details** (OS, Python version)
5. **Steps to reproduce**

---

**Note**: Error messages are designed to be helpful. Read them carefully - they often contain specific guidance about what went wrong and how to fix it.