# 🚀 Quick Start Guide

Get started with A3X in just 5 minutes! This guide will walk you through setting up and running your first autonomous coding agent.

## What is A3X?

A3X is an autonomous coding tool that can understand your goals, write code, run tests, and fix issues automatically. It's like having a coding assistant that never gets tired!

## Prerequisites

- **Python 3.10 or higher**
- **Git** (for cloning repositories)
- **Internet connection** (for AI model access)

## Step 1: Set Up Your Environment (2 minutes)

1. **Clone or download A3X:**
   ```bash
   git clone <repository-url>
   cd A3X
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install A3X:**
   ```bash
   pip install -e .
   ```

## Step 2: Get Your API Key (1 minute)

A3X uses AI models to understand and write code. You'll need an API key:

1. **Sign up at [OpenRouter.ai](https://openrouter.ai/)**
2. **Go to "Keys" in your dashboard**
3. **Create a new API key**
4. **Set it as an environment variable:**
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

   Or create a `.env` file in the project root:
   ```
   OPENROUTER_API_KEY=your-key-here
   ```

## Step 3: Run Your First Task (2 minutes)

Now let's give A3X a simple task to demonstrate its capabilities:

```bash
a3x run --goal "Create a simple calculator that adds two numbers" --config configs/sample.yaml
```

That's it! A3X will:
1. ✅ Plan the implementation
2. ✅ Create the calculator code
3. ✅ Test that it works
4. ✅ Fix any issues automatically

## What Just Happened?

A3X just demonstrated the power of autonomous coding:

- **Planning**: It understood your goal and created a step-by-step plan
- **Implementation**: It wrote the actual code based on the plan
- **Testing**: It ran tests to verify everything works
- **Self-correction**: If something went wrong, it would fix it automatically

## Try These Next

Now that you have the basics, try these more interesting tasks:

```bash
# Create a web API
a3x run --goal "Build a REST API for a todo list" --config configs/sample.yaml

# Fix bugs in existing code
a3x run --goal "Fix the failing tests in my project" --config configs/sample.yaml

# Add new features
a3x run --goal "Add user authentication to my web app" --config configs/sample.yaml

# Refactor and improve
a3x run --goal "Improve the performance of my database queries" --config configs/sample.yaml
```

## Need Help?

- **Stuck?** Check the [Troubleshooting Guide](../troubleshooting/common-issues.md)
- **Want to learn more?** Read the [FAQ](../faq.md)
- **Ready for production?** See the [Operations Guide](../operations/deployment.md)

## What's Next?

You've just experienced autonomous coding! A3X can handle much more complex tasks:

- Building full applications from scratch
- Debugging and fixing existing code
- Refactoring and optimizing codebases
- Writing tests and documentation
- And much more!

The more you use A3X, the better it gets at understanding your coding style and preferences.

---

**🎉 Congratulations!** You've successfully run your first autonomous coding session. Welcome to the future of software development!