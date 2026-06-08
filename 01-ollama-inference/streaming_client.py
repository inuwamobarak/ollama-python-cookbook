"""
Stream tokens to the terminal in real-time.

Prerequisites:
    pip install ollama
    ollama pull llama3.2
"""

import sys
import ollama


def stream_chat(user_message: str, model: str = "llama3.2") -> str:
    """Stream tokens and return the full assembled response."""
    messages = [{"role": "user", "content": user_message}]
    full_response = ""

    print(f"[{model}] ", end="", flush=True)

    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_response += token

    print()  # newline after stream ends
    return full_response


def stream_generate(prompt: str, model: str = "llama3.2") -> str:
    """Stream a raw generation response."""
    full_response = ""

    for chunk in ollama.generate(model=model, prompt=prompt, stream=True):
        token = chunk["response"]
        print(token, end="", flush=True)
        sys.stdout.flush()
        full_response += token

    print()
    return full_response


if __name__ == "__main__":
    print("=== Streaming chat ===")
    stream_chat("Write a haiku about machine learning.")

    print("\n=== Streaming generate ===")
    stream_generate("List 3 reasons Python is popular for AI:")