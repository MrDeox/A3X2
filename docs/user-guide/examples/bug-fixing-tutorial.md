# 🐛 Debugging and Fixing Code with A3X

This tutorial demonstrates how to use A3X for debugging issues and fixing bugs in existing code.

## Scenario: Calculator Bug

Imagine you have a simple calculator that should add numbers, but it's producing wrong results. Let's see how A3X can help debug and fix this.

### Starting Code

```python
# calculator.py
def add_numbers(a, b):
    """Add two numbers together"""
    result = a + b
    return result

def multiply_numbers(a, b):
    """Multiply two numbers"""
    result = a * b
    return result

# Test the calculator
if __name__ == "__main__":
    print(add_numbers(5, 3))    # Should print 8
    print(multiply_numbers(4, 7))  # Should print 28
```

**The bug:** When you run this, `add_numbers(5, 3)` returns `53` instead of `8`.

## Step 1: Problem Analysis

Let's ask A3X to analyze the issue:

```bash
a3x run --goal "Analyze the calculator.py file and identify why add_numbers(5, 3) returns 53 instead of 8" --config configs/sample.yaml
```

**What A3X might discover:**
- The issue is likely with string concatenation vs. numeric addition
- Variable `a` or `b` might be strings instead of numbers
- There could be an import or type conversion issue

## Step 2: Deep Dive Investigation

Let's get more detailed analysis:

```bash
a3x run --goal "Examine the add_numbers function and identify the exact cause of the bug with debugging information" --config configs/sample.yaml
```

**What A3X will do:**
- Add debug prints to understand variable types
- Test the function with different input types
- Identify the root cause (likely string concatenation)

## Step 3: Implement the Fix

Once A3X identifies the issue, let's fix it:

```bash
a3x run --goal "Fix the bug in add_numbers function - ensure proper numeric addition" --config configs/sample.yaml
```

**What A3X will fix:**
- Convert inputs to proper numeric types
- Handle both string and numeric inputs safely
- Add input validation

**Expected fix:**
```python
def add_numbers(a, b):
    """Add two numbers together"""
    # Convert to float to handle both int and string inputs
    try:
        num_a = float(a)
        num_b = float(b)
        result = num_a + num_b
        return result
    except (ValueError, TypeError):
        raise ValueError("Both inputs must be numbers")
```

## Step 4: Add Tests

Let's ensure this bug doesn't happen again:

```bash
a3x run --goal "Create comprehensive tests for the calculator functions including edge cases and the previously broken scenario" --config configs/sample.yaml
```

**What A3X will create:**
- Unit tests for all functions
- Edge case testing (strings, floats, invalid inputs)
- Regression tests for the specific bug

## Step 5: Verify the Fix

Let's run the tests to confirm everything works:

```bash
a3x run --goal "Run the test suite and verify that all calculator functions work correctly" --config configs/sample.yaml
```

**Expected results:**
- All tests pass
- `add_numbers(5, 3)` returns `8`
- Error handling works for invalid inputs

## Advanced Debugging Scenario

### Scenario: Performance Issue

**Problem:** A web API endpoint is very slow.

**Debugging steps:**

1. **Identify the bottleneck:**
   ```bash
   a3x run --goal "Analyze API performance and identify why /users endpoint is slow" --config configs/sample.yaml
   ```

2. **Profile the code:**
   ```bash
   a3x run --goal "Add performance profiling to identify slow database queries or algorithms" --config configs/sample.yaml
   ```

3. **Optimize the code:**
   ```bash
   a3x run --goal "Optimize the slow database queries and improve API response time" --config configs/sample.yaml
   ```

4. **Verify improvement:**
   ```bash
   a3x run --goal "Add benchmarking tests to verify performance improvements" --config configs/sample.yaml
   ```

## Debugging Best Practices with A3X

### 1. Be Specific About the Problem

**❌ Too vague:**
```bash
a3x run --goal "Fix the bug" --config configs/sample.yaml
```

**✅ Specific:**
```bash
a3x run --goal "Fix the bug where login fails with 'Invalid credentials' error" --config configs/sample.yaml
```

### 2. Include Context

**❌ Missing context:**
```bash
a3x run --goal "Debug the error" --config configs/sample.yaml
```

**✅ With context:**
```bash
a3x run --goal "Debug the 'Connection timeout' error that occurs when saving large files" --config configs/sample.yaml
```

### 3. Ask for Comprehensive Solutions

**❌ Just fix:**
```bash
a3x run --goal "Fix the bug" --config configs/sample.yaml
```

**✅ Fix and prevent:**
```bash
a3x run --goal "Fix the bug and add proper error handling to prevent similar issues" --config configs/sample.yaml
```

### 4. Use Step-by-Step Debugging

For complex issues, break it down:

```bash
# Step 1: Understand the problem
a3x run --goal "Analyze the error symptoms and identify possible causes" --config configs/sample.yaml

# Step 2: Gather evidence
a3x run --goal "Add logging and debugging to trace the issue" --config configs/sample.yaml

# Step 3: Implement fix
a3x run --goal "Fix the identified issue" --config configs/sample.yaml

# Step 4: Verify solution
a3x run --goal "Test the fix thoroughly and add regression tests" --config configs/sample.yaml
```

## Common Bug Patterns A3X Can Fix

### 1. Type-Related Bugs

**Problem:** Mixing strings and numbers
```python
# Buggy code
result = "Hello " + 42  # TypeError

# Fixed by A3X
result = "Hello " + str(42)  # Works correctly
```

### 2. Logic Errors

**Problem:** Incorrect conditional logic
```python
# Buggy code
if user.age > 18:
    can_vote = True  # Wrong logic

# Fixed by A3X
if user.age >= 18:
    can_vote = True  # Correct logic
```

### 3. Exception Handling

**Problem:** Unhandled errors crash the application
```python
# Buggy code
def divide(a, b):
    return a / b  # ZeroDivisionError possible

# Fixed by A3X
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Cannot divide by zero"
```

### 4. Import Issues

**Problem:** Missing or incorrect imports
```python
# Buggy code
import non_existent_module  # ImportError

# Fixed by A3X
from datetime import datetime  # Correct import
```

## Testing After Bug Fixes

### Always Verify Fixes

```bash
a3x run --goal "Run comprehensive tests to verify the bug fix works and doesn't break existing functionality" --config configs/sample.yaml
```

### Add Regression Tests

```bash
a3x run --goal "Add regression tests for the specific bug that was fixed to prevent it from reoccurring" --config configs/sample.yaml
```

## When A3X Can't Find the Bug

### Provide More Context

If A3X struggles to identify the issue:

1. **Share error messages:**
   ```bash
   a3x run --goal "Debug this error: 'TypeError: unsupported operand type(s) for +: 'int' and 'str''" --config configs/sample.yaml
   ```

2. **Point to specific code:**
   ```bash
   a3x run --goal "Debug the issue in the calculate_total function where string and int are being added" --config configs/sample.yaml
   ```

3. **Explain expected vs. actual behavior:**
   ```bash
   a3x run --goal "Fix the bug where add_numbers(5, 3) returns 53 instead of 8" --config configs/sample.yaml
   ```

## Real-World Example: Database Connection Bug

### Scenario
A web application fails to connect to the database intermittently.

**Debugging workflow:**

1. **Analyze connection code:**
   ```bash
   a3x run --goal "Analyze database connection code and identify potential connection issues" --config configs/sample.yaml
   ```

2. **Add error handling:**
   ```bash
   a3x run --goal "Add proper error handling and retry logic for database connections" --config configs/sample.yaml
   ```

3. **Create connection tests:**
   ```bash
   a3x run --goal "Add tests for database connectivity and error scenarios" --config configs/sample.yaml
   ```

4. **Add monitoring:**
   ```bash
   a3x run --goal "Add logging to monitor database connection health" --config configs/sample.yaml
   ```

## Summary

A3X excels at:

- **Identifying bugs** through code analysis
- **Understanding error patterns** and root causes
- **Implementing fixes** with proper error handling
- **Adding tests** to prevent regression
- **Improving code quality** overall

**Key takeaway:** A3X isn't just a code generator - it's a debugging partner that can analyze, understand, and fix complex issues in your codebase.

---

**Ready to try debugging?** Start with a specific bug description and let A3X show you its analytical capabilities!