"""
Basic text generation with the official ollama-python SDK.

Prerequisites:
    pip install ollama
    ollama pull llama3.2
"""

import ollama


# ── 1. Simple one-shot generation ────────────────────────────────────────────
def generate(prompt: str, model: str = "llama3.2") -> str:
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"]


# ── 2. Chat with a system prompt ─────────────────────────────────────────────
def chat(user_message: str, model: str = "llama3.2") -> str:
    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": user_message},
    ]
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


# ── 3. List locally available models ─────────────────────────────────────────
def list_models() -> list[str]:
    return [m["name"] for m in ollama.list()["models"]]


if __name__ == "__main__":
    print("=== generate() ===")
    print(generate("Why is the sky blue? Answer in one sentence."))

    print("\n=== chat() ===")
    print(chat("What is the capital of France?"))

    print("\n=== Available models ===")
    print(list_models())