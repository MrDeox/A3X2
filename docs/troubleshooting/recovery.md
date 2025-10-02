# 🔄 Recovery Procedures

This guide provides step-by-step procedures for recovering from various failure scenarios and getting A3X back to working condition.

## Quick Recovery Checklist

Before diving into specific procedures, try this quick checklist:

1. **Restart A3X** - Sometimes a simple restart fixes transient issues
2. **Check API key** - Verify `OPENROUTER_API_KEY` is set correctly
3. **Test connectivity** - Ensure internet connection works
4. **Check disk space** - Verify adequate storage is available
5. **Review logs** - Look for specific error messages

## Recovery Scenario 1: API Key Issues

### Symptoms
- "Invalid API key" errors
- Authentication failures
- "Rate limit exceeded" messages

### Recovery Steps

1. **Verify current API key**:
   ```bash
   echo $OPENROUTER_API_KEY | head -c 10  # Should show key prefix
   ```

2. **Check key validity**:
   - Visit [OpenRouter dashboard](https://openrouter.ai/keys)
   - Verify key exists and has credits

3. **Generate new key if needed**:
   - Create new key in dashboard
   - Update environment variable:
     ```bash
     export OPENROUTER_API_KEY="new-key-here"
     echo "OPENROUTER_API_KEY=new-key-here" > .env
     ```

4. **Test new key**:
   ```bash
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        https://openrouter.ai/api/v1/models | head -5
   ```

5. **Retry A3X operation**:
   ```bash
   a3x run --goal "test recovery" --config configs/sample.yaml
   ```

## Recovery Scenario 2: Configuration Corruption

### Symptoms
- YAML parsing errors
- Configuration file not found
- Settings not taking effect

### Recovery Steps

1. **Backup current config**:
   ```bash
   cp configs/sample.yaml configs/sample.yaml.backup
   ```

2. **Validate config syntax**:
   ```bash
   python -c "import yaml; yaml.safe_load(open('configs/sample.yaml'))"
   ```

3. **Restore from backup if needed**:
   ```bash
   cp configs/sample.yaml.backup configs/sample.yaml
   ```

4. **Test with minimal config**:
   ```bash
   a3x run --goal "simple test" --config configs/manual.yaml
   ```

5. **Gradually restore settings**:
   - Copy working sections from backup
   - Test after each change

## Recovery Scenario 3: Code Generation Failures

### Symptoms
- Generated code has syntax errors
- Import failures in generated code
- Test failures after code generation

### Recovery Steps

1. **Review generated code**:
   - Check for obvious syntax issues
   - Verify import statements

2. **Ask A3X to self-correct**:
   ```bash
   a3x run --goal "Fix the syntax errors in the generated code" --config configs/sample.yaml
   ```

3. **Run tests manually**:
   ```bash
   python -m pytest tests/ -v  # Identify specific failures
   ```

4. **Provide more context**:
   ```bash
   a3x run --goal "Fix the failing tests with proper error handling" --config configs/sample.yaml
   ```

5. **Try different approach**:
   ```bash
   a3x run --goal "Rewrite the problematic function with better error handling" --config configs/sample.yaml
   ```

## Recovery Scenario 4: Performance Degradation

### Symptoms
- Operations taking much longer than usual
- Memory usage constantly high
- Frequent timeouts or crashes

### Recovery Steps

1. **Check system resources**:
   ```bash
   top -o %MEM | head -10  # Check memory usage
   df -h                   # Check disk space
   ```

2. **Clear temporary files**:
   ```bash
   find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
   find . -name "*.pyc" -delete
   find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null
   ```

3. **Optimize configuration**:
   ```yaml
   # In configs/sample.yaml
   history:
     max_context_length: 2000  # Reduce context size
   execution:
     command_timeout: 120      # Adjust timeouts
   ```

4. **Restart virtual environment**:
   ```bash
   deactivate
   source .venv/bin/activate
   ```

5. **Monitor performance**:
   ```bash
   a3x run --goal "test performance" --config configs/sample.yaml
   ```

## Recovery Scenario 5: Data Corruption

### Symptoms
- Files modified unexpectedly
- Loss of important code or data
- Inconsistent project state

### Recovery Steps

1. **Check git status** (if using git):
   ```bash
   git status
   git diff
   ```

2. **Restore from backup**:
   ```bash
   git checkout HEAD -- path/to/damaged/file
   # Or restore from external backup
   ```

3. **Verify project integrity**:
   ```bash
   python -m pytest tests/ -x  # Stop on first failure
   ```

4. **Review recent changes**:
   ```bash
   a3x run --goal "Review and fix any issues in the codebase" --config configs/sample.yaml
   ```

## Recovery Scenario 6: Environment Issues

### Symptoms
- Installation broken
- Dependencies missing or corrupted
- Path or permission issues

### Recovery Steps

1. **Check Python environment**:
   ```bash
   which python
   python --version
   ```

2. **Reinstall dependencies**:
   ```bash
   pip uninstall a3x
   pip install -e .
   ```

3. **Fix environment issues**:
   ```bash
   # Recreate virtual environment if needed
   rm -rf .venv
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

4. **Test basic functionality**:
   ```bash
   python -c "import a3x; print('Import successful')"
   a3x --help
   ```

## Advanced Recovery Techniques

### Creating Restore Points

**Before major operations**:
```bash
# Git users
git add .
git commit -m "Before major A3X operation"

# Backup important files
cp -r project/ project-backup-$(date +%Y%m%d-%H%M%S)/
```

### Using Safe Mode

**Run A3X in safe mode**:
```bash
a3x run --goal "careful operation" --config configs/sample.yaml --dry-run --max-steps 1
```

### Recovery from Complete Failure

**If A3X is completely broken**:

1. **Start fresh**:
   ```bash
   # Backup important data
   cp -r seed/ important-data-backup/
   cp -r project/ project-backup/

   # Remove and reinstall
   pip uninstall a3x
   rm -rf .venv/
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Restore data**:
   ```bash
   cp -r important-data-backup/* seed/
   cp -r project-backup/* project/
   ```

## Prevention Measures

### Regular Maintenance

1. **Keep backups**:
   - Regular git commits
   - Periodic project snapshots

2. **Monitor resources**:
   ```bash
   # Add to cron for monitoring
   df -h | grep -v "tmpfs\|devtmpfs" > disk_usage.log
   ```

3. **Update regularly**:
   ```bash
   pip install --upgrade a3x
   ```

### Configuration Management

1. **Use version control for configs**:
   ```bash
   git add configs/
   git commit -m "Update configuration"
   ```

2. **Document configuration changes**:
   - Keep notes about why settings were changed

3. **Test config changes**:
   ```bash
   a3x run --goal "test config changes" --config configs/sample.yaml
   ```

## Getting Help After Recovery

### Verify Recovery Success

After following recovery procedures:

1. **Test basic functionality**:
   ```bash
   a3x run --goal "simple test" --config configs/manual.yaml
   ```

2. **Test with original goal**:
   ```bash
   a3x run --goal "your-original-goal" --config configs/sample.yaml
   ```

3. **Monitor for issues**:
   - Watch for error patterns
   - Check resource usage

### When to Seek Additional Help

**Seek help if**:
- Recovery procedures don't work
- Issues keep recurring
- You're unsure about data integrity
- Performance problems persist

**Provide when asking for help**:
- Error messages and logs
- Steps already tried
- Configuration files
- System information

---

**Remember**: Most issues are recoverable. A3X is designed to be resilient, and these procedures should get you back to productive work quickly.