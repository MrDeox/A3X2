# ❓ Frequently Asked Questions

This FAQ addresses common questions about A3X, autonomous coding, and the SeedAI approach.

## Getting Started

### What is A3X?

A3X is an autonomous coding tool that uses artificial intelligence to understand your goals, write code, run tests, and fix issues automatically. It's like having a coding assistant that never gets tired and can work 24/7.

### Do I need coding experience to use A3X?

**No!** A3X is designed for everyone:
- **Non-coders**: Describe what you want in plain English
- **Junior developers**: Get help with syntax and best practices
- **Senior developers**: Automate repetitive tasks and get suggestions
- **Teams**: Maintain consistent code quality and speed up development

### How is A3X different from other coding tools?

| Feature | A3X | Traditional AI Coding | Human Developers |
|---------|-----|---------------------|------------------|
| **Autonomous** | ✅ Fully autonomous | ❌ Requires guidance | ❌ Manual work |
| **24/7 Available** | ✅ Never sleeps | ✅ Sometimes | ❌ Limited hours |
| **Consistent Quality** | ✅ No fatigue | ✅ Consistent | ⚠️ Variable |
| **Learning Ability** | ✅ Improves over time | ❌ Static | ✅ Can learn |
| **Safety** | ✅ Built-in safeguards | ⚠️ Varies | ⚠️ Human error |

## Setup & Installation

### Which operating system does A3X support?

A3X works on:
- **Linux** (Ubuntu, Debian, CentOS, etc.)
- **macOS** (10.15+)
- **Windows** (via WSL or native Python)

### Do I need a powerful computer?

**Minimum requirements:**
- 4GB RAM (8GB recommended)
- 2GB free disk space
- Internet connection for AI models

**Performance tips:**
- More RAM = better performance with large codebases
- SSD drive = faster file operations
- Stable internet = reliable AI responses

### Can I use A3X offline?

Currently, A3X requires internet for AI model access. Future versions may support local models via Ollama or similar tools.

## Usage & Operation

### What kinds of tasks can A3X handle?

A3X excels at:

**✅ Great for:**
- Building applications from scratch
- Debugging and fixing existing code
- Refactoring and improving code
- Writing tests and documentation
- Setting up project configurations

**⚠️ Challenging for:**
- GUI design (can generate code, but UI feedback is limited)
- Real-time systems (timing-sensitive applications)
- Hardware integration (requires specific drivers/knowledge)

**💡 Examples:**
```bash
# Build a web API
a3x run --goal "Create a REST API for a blog platform" --config configs/sample.yaml

# Fix bugs
a3x run --goal "Debug why user login is failing" --config configs/sample.yaml

# Add features
a3x run --goal "Add email notifications to user registration" --config configs/sample.yaml
```

### How do I write effective goals for A3X?

**Tips for better results:**

1. **Be specific**:
   ```
   ❌ "Make it better"
   ✅ "Add error handling to the calculate_total function"
   ```

2. **Include context**:
   ```
   ❌ "Create login system"
   ✅ "Create user authentication with email/password and JWT tokens"
   ```

3. **Break down complex tasks**:
   ```bash
   a3x run --goal "Step 1: Create user model with validation" --config configs/sample.yaml
   a3x run --goal "Step 2: Add login/logout endpoints" --config configs/sample.yaml
   ```

4. **Specify technologies**:
   ```
   ❌ "Build a website"
   ✅ "Build a React web app with Node.js backend"
   ```

### Why do I get different results each time?

A3X uses AI models that can be creative. For consistent results:

- **Use specific goals** (see above)
- **Provide examples** of desired output style
- **Review and iterate** based on results
- **Use configuration** to set preferences

### Is A3X safe? Can it break my code?

**Safety features:**
- Incremental changes (not complete rewrites)
- Automatic testing after each change
- Dry-run mode for safe testing
- Undo capabilities for recent changes

**Best practices:**
- Start with `--dry-run` for new projects
- Use version control (git)
- Review changes before accepting
- Test thoroughly after major operations

## Cost & Pricing

### How much does A3X cost to use?

**Current model:**
- A3X itself is free and open source
- You pay for AI model usage (via OpenRouter)
- Typical costs: $0.10 - $2.00 per hour of active use

**Cost factors:**
- Model choice (faster models cost more)
- Project complexity (larger codebases use more tokens)
- Operation frequency (continuous use vs. occasional)

### Can I use free AI models?

Yes! Options include:
- **OpenRouter free tiers** (limited usage)
- **Local models** (via Ollama, future support)
- **Manual mode** (pre-defined scripts, no AI costs)

## Technical Questions

### What programming languages does A3X support?

A3X works with any text-based programming language:

**🌟 Best support:**
- Python (most common use case)
- JavaScript/TypeScript
- Java
- C#
- Go

**✅ Also works with:**
- C/C++
- Rust
- PHP
- Ruby
- Shell scripts
- Configuration files (YAML, JSON, etc.)

**💡 Language-specific features:**
- Automatic dependency management
- Framework-specific patterns
- Best practice implementation

### Can A3X work with my existing codebase?

Yes! A3X is designed to work with existing projects:

**Capabilities:**
- Analyze existing code structure
- Understand current patterns and conventions
- Make changes while preserving style
- Add features without breaking existing functionality

**Best practices:**
- Start with small, specific goals
- Let A3X learn your codebase patterns
- Review changes to ensure compatibility

### What are "seeds" in A3X?

**Seeds are improvement ideas:**

```yaml
# Example seed
seed:
  name: "add-user-authentication"
  goal: "Implement secure user login system"
  priority: "high"
  requirements: "email/password validation, JWT tokens"
  success_criteria: "users can register and login securely"
```

**How seeds work:**
1. **Detection**: A3X identifies improvement opportunities
2. **Prioritization**: Seeds are ranked by potential impact
3. **Execution**: High-priority seeds are executed automatically
4. **Learning**: Results inform future improvements

## Troubleshooting

### Why does A3X say "Goal not understood"?

**Common causes:**
- Goal is too vague or abstract
- Missing technical details
- Unclear success criteria

**Solutions:**
- Add specific requirements
- Include technical constraints
- Provide examples of desired outcome

### "Rate limit exceeded" - what does this mean?

**Rate limiting happens when:**
- Too many API requests sent too quickly
- Using free tier with usage limits
- Model is experiencing high demand

**Solutions:**
- Wait a few minutes and retry
- Use a different model with higher limits
- Upgrade to paid OpenRouter plan

### A3X generated code with errors - what now?

**Don't worry!** This is normal:

1. **A3X can fix its own errors**:
   ```bash
   a3x run --goal "Fix the syntax errors in the generated code" --config configs/sample.yaml
   ```

2. **Review and provide feedback**:
   - Tell A3X what went wrong
   - Ask for corrections with more context

3. **Try different approach**:
   ```bash
   a3x run --goal "Rewrite the function using a different approach" --config configs/sample.yaml
   ```

## Advanced Usage

### How do I customize A3X behavior?

**Configuration options:**
- **Model selection**: Choose different AI models
- **Safety settings**: Adjust risk tolerance
- **Performance tuning**: Optimize for speed vs. quality
- **Custom prompts**: Define coding style preferences

**Example custom config:**
```yaml
llm:
  model: "anthropic/claude-3-sonnet"
  temperature: 0.3  # More focused responses

execution:
  max_iterations: 20
  command_timeout: 300

safety:
  allow_network: false
  require_tests: true
```

### Can A3X learn from my preferences?

**Yes!** A3X improves over time:

- **Pattern recognition**: Learns your coding style
- **Preference adaptation**: Adjusts to your feedback
- **Context awareness**: Remembers project structure
- **Self-improvement**: Uses seeds to enhance capabilities

**To help A3X learn:**
- Provide clear feedback on results
- Use consistent goal phrasing
- Review and rate outputs when possible

### What's the difference between "run" and "seed run"?

| Command | Purpose | Use Case |
|---------|---------|----------|
| `a3x run` | Execute single goal | One-time tasks, experiments |
| `a3x seed run` | Execute improvement seeds | Ongoing development, maintenance |

**Examples:**
```bash
# Single task
a3x run --goal "Add user registration" --config configs/sample.yaml

# Autonomous improvement
a3x seed run --config configs/seed_manual.yaml
```

## Future & Development

### Will A3X replace human developers?

**No!** A3X is designed to augment, not replace:

**A3X excels at:**
- Repetitive coding tasks
- Following established patterns
- Quick prototyping
- 24/7 availability

**Humans are better at:**
- Creative problem solving
- Understanding user needs
- Making architectural decisions
- Complex debugging

**Best workflow:**
1. Human defines goals and requirements
2. A3X implements solutions
3. Human reviews and provides feedback
4. A3X iterates and improves

### How often is A3X updated?

A3X follows continuous improvement:

- **Regular updates** with bug fixes and features
- **Community contributions** welcome
- **Seed-driven evolution** for autonomous improvements

### Can I contribute to A3X development?

**Absolutely!** Contributions are welcome:

- **Bug reports**: Help identify and fix issues
- **Feature requests**: Suggest new capabilities
- **Code contributions**: Improve existing functionality
- **Documentation**: Help others learn A3X

## Support & Community

### Where can I get help?

**Resources available:**
- **Documentation**: This comprehensive guide
- **Troubleshooting**: [Common Issues Guide](troubleshooting/common-issues.md)
- **Community forums**: GitHub discussions
- **Issue tracker**: Bug reports and feature requests

### How do I report bugs or request features?

**For bugs:**
1. Check if already reported
2. Create detailed issue with:
   - Steps to reproduce
   - Error messages
   - Environment details
   - Expected vs. actual behavior

**For features:**
1. Search existing requests
2. Create feature request with:
   - Use case description
   - Proposed solution
   - Benefits and impact

## Common Misconceptions

### "A3X will write perfect code every time"

**Reality:** A3X aims for good, working code that may need refinement:

- First attempts might need adjustments
- Complex requirements may need multiple iterations
- Domain-specific knowledge might be required

### "A3X is just another code generator"

**Beyond code generation:**
- **Testing**: Automatically verifies code works
- **Debugging**: Identifies and fixes issues
- **Learning**: Improves based on feedback
- **Safety**: Includes safeguards and validation

### "A3X requires constant supervision"

**Actually:** A3X is designed for autonomy:

- Works independently once given clear goals
- Handles errors and edge cases automatically
- Can run for extended periods unsupervised
- Provides progress updates and summaries

---

**Still have questions?** Check the [Troubleshooting Guide](troubleshooting/common-issues.md) or search existing documentation for more specific help.