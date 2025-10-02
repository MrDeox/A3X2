# 🎯 Your First Autonomous Run

This guide walks you through your first experience with A3X's autonomous coding capabilities.

## What to Expect

In your first run, you'll see A3X:
- Analyze your goal and create a plan
- Write or modify code to achieve the goal
- Run tests to verify the solution works
- Fix any issues automatically
- Provide a summary of what was accomplished

## Before You Start

Make sure you have:
- ✅ Completed the [installation](installation.md)
- ✅ Set up your [API key](../user-guide/configuration.md#api-keys)
- ✅ Activated your virtual environment

## Your First Run

Let's start with something simple but impressive:

```bash
a3x run --goal "Create a simple calculator that adds two numbers" --config configs/sample.yaml
```

### What You'll See

1. **Initialization**: A3X loads and shows configuration
2. **Planning**: The AI analyzes your request and creates a step-by-step plan
3. **Implementation**: Code is written or modified
4. **Testing**: Tests run to verify everything works
5. **Self-correction**: If issues arise, A3X fixes them automatically
6. **Completion**: You'll get a summary of what was accomplished

## Understanding the Output

During the run, you'll see:

```
[INFO] Starting autonomous run...
[PLAN] Creating a simple calculator with addition functionality
[ACTION] Creating calculator.py with add function
[TEST] Running tests to verify calculator works
[RESULT] Calculator successfully created and tested!
```

### Real-time Monitoring

A3X shows you exactly what's happening at each step, so you can:
- See the planning process
- Monitor code changes
- Watch tests execute
- Understand any issues and fixes

## Try These Beginner-Friendly Examples

### Example 1: Simple Utility
```bash
a3x run --goal "Create a script that converts temperatures between Celsius and Fahrenheit" --config configs/sample.yaml
```

### Example 2: Data Processing
```bash
a3x run --goal "Create a program that reads a CSV file and calculates statistics" --config configs/sample.yaml
```

### Example 3: Text Processing
```bash
a3x run --goal "Build a tool that counts words and characters in text files" --config configs/sample.yaml
```

## What If Something Goes Wrong?

Don't worry! A3X is designed to handle issues gracefully:

- **API Issues**: Check your internet connection and API key
- **Code Errors**: A3X will attempt to fix them automatically
- **Test Failures**: A3X will analyze and correct the issues

If you're still stuck, check the [Troubleshooting Guide](../troubleshooting/common-issues.md).

## Understanding Autonomous Behavior

### How A3X Makes Decisions

A3X uses AI to:
1. **Understand** your goal in natural language
2. **Plan** the implementation steps
3. **Execute** code changes safely
4. **Verify** that changes work correctly
5. **Learn** from each interaction

### Safety First

A3X prioritizes safety:
- Code changes are made incrementally
- Tests run after each change
- Unsafe operations are blocked by default
- You can interrupt at any time with Ctrl+C

## Your Role

As the user, you:
- **Define goals** in plain English
- **Review results** when complete
- **Provide feedback** for continuous improvement
- **Monitor progress** during long-running tasks

## Next Steps

After your first successful run:

1. **Review the generated code** and understand what was created
2. **Try a more complex goal** to see A3X's capabilities
3. **Explore different types of tasks** (web apps, data processing, etc.)
4. **Learn about advanced features** in the [User Guide](../user-guide/concepts.md)

## Getting Help

- **Command help**: `a3x run --help`
- **Configuration help**: See [Configuration Guide](../user-guide/configuration.md)
- **Troubleshooting**: [Common Issues](../troubleshooting/common-issues.md)
- **Examples**: [Practical Examples](../user-guide/examples/)

## Advanced First Runs

Once you're comfortable with basic runs, try:

```bash
# Work with existing code
a3x run --goal "Fix the bugs in my existing project" --config configs/sample.yaml

# Build something more complex
a3x run --goal "Create a web API with user authentication" --config configs/sample.yaml

# Refactor and improve
a3x run --goal "Optimize my code for better performance" --config configs/sample.yaml
```

---

**🎉 Congratulations!** You've completed your first autonomous coding session. You now understand how A3X works and can start using it for real development tasks.