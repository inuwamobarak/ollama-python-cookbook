# Contributing to ollama-python-cookbook

Thank you for wanting to improve this project! Here's how to do it well.

## Adding a new recipe

1. **Fork** the repo and create a branch: `git checkout -b add-my-recipe`

2. **Choose the right folder** — or propose a new one if it truly doesn't fit.

3. **Follow the script template:**

```python
"""
One-line description of what this demonstrates.

Prerequisites:
    pip install package-a package-b
    ollama pull model-name
"""
# ... type-hinted, idiomatic Python ...

if __name__ == "__main__":
    # A working demo that a new user can run immediately
    pass
```

4. **Rules for acceptance:**
   - Must run end-to-end without modification (besides having Ollama installed)
   - Must use type hints
   - Must have a docstring and a `__main__` demo block
   - No hardcoded absolute paths
   - Handles errors — no bare `except:` clauses

5. **Open a PR** with a title like `feat: add llm-as-judge pattern to 07-agents`

## Reporting a bug

Open an issue with:
- The script name and section
- Your OS and Python version
- The exact error message
- The command you ran

## Questions

Open a Discussion — not an Issue — for general questions.