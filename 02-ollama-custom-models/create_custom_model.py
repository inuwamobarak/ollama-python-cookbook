"""
Build and register a custom model via Python.

Prerequisites:
    pip install ollama
    ollama pull llama3.2
"""

import ollama


# ── Define Modelfile as a Python string ──────────────────────────────────
MODELFILE = """
FROM llama3.2

# Lower temperature = more focused, deterministic responses
PARAMETER temperature 0.3
PARAMETER top_p 0.85

SYSTEM \"\"\"
You are a senior Python engineer who writes clean, idiomatic code.
Rules:
- Always use type hints
- Prefer pathlib over os.path
- Follow PEP 8 strictly
- Add concise docstrings to every function
- Never use bare except clauses
\"\"\"
"""


def create_model(name: str, modelfile: str) -> None:
    """Create or overwrite a custom Ollama model."""
    print(f"Creating model '{name}'...")

    # Stream progress so long builds don't look frozen
    for progress in ollama.create(model=name, modelfile=modelfile, stream=True):
        status = progress.get("status", "")
        if status:
            print(f"  {status}")

    print(f"✓ Model '{name}' ready.")


def delete_model(name: str) -> None:
    """Remove a custom model."""
    ollama.delete(name)
    print(f"✓ Model '{name}' deleted.")


def show_model(name: str) -> None:
    """Print the model's system prompt and parameters."""
    info = ollama.show(name)
    print(f"\n=== {name} info ===")
    print(info.get("modelfile", "No modelfile found."))


if __name__ == "__main__":
    MODEL_NAME = "python-expert"

    create_model(MODEL_NAME, MODELFILE)
    show_model(MODEL_NAME)

    # Test the custom model
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Write a function to read a JSON file safely."}],
    )
    print("\n=== Test response ===")
    print(response["message"]["content"])

    # Uncomment to clean up:
    # delete_model(MODEL_NAME)