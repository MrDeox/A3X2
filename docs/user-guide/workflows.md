# 🔄 Common Usage Workflows

This guide shows practical workflows for using A3X effectively in real-world scenarios.

## Workflow 1: New Project Development

**When to use:** Starting a new application from scratch

### Step-by-Step Process

1. **Project Setup**
   ```bash
   a3x run --goal "Initialize Python project with proper structure" --config configs/sample.yaml
   ```

2. **Core Functionality**
   ```bash
   a3x run --goal "Create user management system with authentication" --config configs/sample.yaml
   ```

3. **API Development**
   ```bash
   a3x run --goal "Build REST API endpoints for user operations" --config configs/sample.yaml
   ```

4. **Testing**
   ```bash
   a3x run --goal "Add comprehensive test suite" --config configs/sample.yaml
   ```

5. **Documentation**
   ```bash
   a3x run --goal "Generate API documentation" --config configs/sample.yaml
   ```

### Expected Timeline
- **Small project**: 30 minutes - 2 hours
- **Medium project**: 2-6 hours
- **Large project**: 6+ hours (may require multiple runs)

## Workflow 2: Bug Investigation and Fixing

**When to use:** Debugging issues in existing code

### Step-by-Step Process

1. **Problem Analysis**
   ```bash
   a3x run --goal "Analyze the login failure and identify root cause" --config configs/sample.yaml
   ```

2. **Code Review**
   ```bash
   a3x run --goal "Review authentication code for potential issues" --config configs/sample.yaml
   ```

3. **Implement Fix**
   ```bash
   a3x run --goal "Fix the identified authentication bug" --config configs/sample.yaml
   ```

4. **Verification**
   ```bash
   a3x run --goal "Test the fix and ensure login works correctly" --config configs/sample.yaml
   ```

5. **Regression Testing**
   ```bash
   a3x run --goal "Run full test suite to ensure no regressions" --config configs/sample.yaml
   ```

### Tips for Bug Fixing
- **Be specific** about the problem symptoms
- **Include error messages** in your goals
- **Specify expected behavior** clearly
- **Ask for tests** to prevent future regressions

## Workflow 3: Feature Enhancement

**When to use:** Adding new features to existing applications

### Step-by-Step Process

1. **Requirements Analysis**
   ```bash
   a3x run --goal "Analyze current codebase and plan email notification feature" --config configs/sample.yaml
   ```

2. **Design Integration**
   ```bash
   a3x run --goal "Design email notification system architecture" --config configs/sample.yaml
   ```

3. **Implementation**
   ```bash
   a3x run --goal "Implement email notification functionality" --config configs/sample.yaml
   ```

4. **Configuration**
   ```bash
   a3x run --goal "Add configuration for email settings" --config configs/sample.yaml
   ```

5. **Testing & Documentation**
   ```bash
   a3x run --goal "Add tests and documentation for email features" --config configs/sample.yaml
   ```

## Workflow 4: Code Quality Improvement

**When to use:** Refactoring and improving existing code

### Step-by-Step Process

1. **Code Analysis**
   ```bash
   a3x run --goal "Analyze code quality and identify improvement opportunities" --config configs/sample.yaml
   ```

2. **Performance Optimization**
   ```bash
   a3x run --goal "Optimize database queries and improve performance" --config configs/sample.yaml
   ```

3. **Code Cleanup**
   ```bash
   a3x run --goal "Refactor code for better readability and maintainability" --config configs/sample.yaml
   ```

4. **Standards Compliance**
   ```bash
   a3x run --goal "Ensure code follows best practices and style guidelines" --config configs/sample.yaml
   ```

## Workflow 5: Testing and Quality Assurance

**When to use:** Improving test coverage and quality

### Step-by-Step Process

1. **Coverage Analysis**
   ```bash
   a3x run --goal "Analyze current test coverage and identify gaps" --config configs/sample.yaml
   ```

2. **Test Generation**
   ```bash
   a3x run --goal "Generate comprehensive tests for uncovered code" --config configs/sample.yaml
   ```

3. **Edge Case Testing**
   ```bash
   a3x run --goal "Add tests for edge cases and error conditions" --config configs/sample.yaml
   ```

4. **Integration Testing**
   ```bash
   a3x run --goal "Create integration tests for API endpoints" --config configs/sample.yaml
   ```

## Workflow 6: Documentation and Knowledge Sharing

**When to use:** Creating and updating documentation

### Step-by-Step Process

1. **Code Documentation**
   ```bash
   a3x run --goal "Generate comprehensive docstrings for all functions" --config configs/sample.yaml
   ```

2. **API Documentation**
   ```bash
   a3x run --goal "Create API documentation with examples" --config configs/sample.yaml
   ```

3. **User Guides**
   ```bash
   a3x run --goal "Generate user guide for the application" --config configs/sample.yaml
   ```

4. **Developer Onboarding**
   ```bash
   a3x run --goal "Create developer onboarding documentation" --config configs/sample.yaml
   ```

## Workflow Tips and Best Practices

### Writing Effective Goals

**❌ Poor goals:**
- "Make it better"
- "Fix stuff"
- "Add things"

**✅ Good goals:**
- "Add email validation to user registration"
- "Fix the memory leak in data processing"
- "Optimize database queries for better performance"

### Managing Long-Running Operations

**For complex tasks:**

1. **Break into phases**:
   ```bash
   a3x run --goal "Phase 1: Analyze requirements for user management" --config configs/sample.yaml
   a3x run --goal "Phase 2: Implement user model and database" --config configs/sample.yaml
   a3x run --goal "Phase 3: Add authentication endpoints" --config configs/sample.yaml
   ```

2. **Monitor progress**:
   - A3X provides real-time updates
   - You can interrupt with Ctrl+C if needed
   - Results are summarized at completion

3. **Handle interruptions**:
   - A3X can resume from interruptions
   - Use checkpoints for very long operations

### Quality Assurance During Workflows

**Always include testing**:
```bash
a3x run --goal "Add feature AND comprehensive tests for the new functionality" --config configs/sample.yaml
```

**Verify after changes**:
```bash
a3x run --goal "Run full test suite and verify no regressions" --config configs/sample.yaml
```

### Handling Workflow Variations

**Different project types:**

| Project Type | Key Considerations | Example Goals |
|-------------|-------------------|---------------|
| **Web API** | Endpoints, validation, security | "Create REST API with JWT auth" |
| **Data Processing** | Performance, error handling | "Build data pipeline with validation" |
| **CLI Tool** | Usability, error messages | "Create CLI with helpful output" |
| **Library** | Documentation, examples | "Build library with full docs" |

**Different team sizes:**

| Team Size | Workflow Adaptation | Communication |
|-----------|-------------------|--------------|
| **Solo Developer** | Full autonomy, detailed goals | Self-documenting code |
| **Small Team** | Coordinate changes, share goals | Clear commit messages |
| **Large Team** | Integration testing, code review | Detailed documentation |

## Troubleshooting Workflows

### When Workflows Fail

1. **Check for common issues**:
   - API connectivity problems
   - Configuration errors
   - Unclear or ambiguous goals

2. **Break down complex workflows**:
   ```bash
   # Instead of one complex goal
   a3x run --goal "Build complete user management system" --config configs/sample.yaml

   # Use multiple focused goals
   a3x run --goal "Create user model" --config configs/sample.yaml
   a3x run --goal "Add user API endpoints" --config configs/sample.yaml
   a3x run --goal "Implement authentication" --config configs/sample.yaml
   ```

3. **Use safe mode for testing**:
   ```bash
   a3x run --goal "test approach" --config configs/sample.yaml --dry-run
   ```

### Recovery Strategies

**If a workflow gets stuck:**

1. **Interrupt and restart**:
   ```bash
   # Stop current operation (Ctrl+C)
   # Start with clearer goal
   a3x run --goal "Complete the user model that was started" --config configs/sample.yaml
   ```

2. **Use manual intervention**:
   ```bash
   a3x run --goal "Fix the issues in the generated code" --config configs/sample.yaml
   ```

3. **Start fresh if needed**:
   ```bash
   a3x run --goal "Rebuild user management with cleaner approach" --config configs/sample.yaml
   ```

## Workflow Automation

### Using Seeds for Recurring Tasks

**Set up automated workflows**:

```yaml
# In seed/backlog.yaml
seeds:
  - name: "weekly-testing"
    goal: "Run full test suite and report results"
    schedule: "weekly"
    priority: "medium"

  - name: "security-audit"
    goal: "Review code for security vulnerabilities"
    schedule: "monthly"
    priority: "high"
```

**Execute recurring workflows**:
```bash
a3x seed run --config configs/seed_manual.yaml
```

### Integration with Development Tools

**CI/CD Integration**:
```bash
# In your CI pipeline
a3x run --goal "Run tests and verify code quality" --config configs/sample.yaml
```

**Git Hooks**:
```bash
# Pre-commit hook
a3x run --goal "Check code style and run basic tests" --config configs/sample.yaml
```

---

**Remember:** These workflows are flexible starting points. Adapt them to your specific needs, project requirements, and team preferences. The key is starting with clear goals and iterating based on results.