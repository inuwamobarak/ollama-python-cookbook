"""
Use the official openai package against local Ollama.

This is useful when migrating from OpenAI to local models — no code changes
needed except the base_url and model name.

Prerequisites:
    pip install openai
    ollama pull llama3.2
"""

from openai import OpenAI

# Point the OpenAI client at local Ollama instance
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by the client, but ignored by Ollama
)

MODEL = "llama3.2"


# ── 1. Basic chat completion ──────────────────────────────────────────────────
def chat(message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message.content


# ── 2. Streaming chat ─────────────────────────────────────────────────────────
def stream_chat(message: str) -> str:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": message}],
        stream=True,
    )
    full = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        print(token, end="", flush=True)
        full += token
    print()
    return full


# ── 3. Embeddings ─────────────────────────────────────────────────────────────
def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="nomic-embed-text",
        input=text,
    )
    return response.data[0].embedding


if __name__ == "__main__":
    print("=== Standard chat ===")
    print(chat("What is Ollama?"))

    print("\n=== Streaming chat ===")
    stream_chat("List 3 open-source LLMs in one sentence each.")

    print("\n=== Embedding dimensions ===")
    vec = embed("Hello, world!")
    print(f"Vector length: {len(vec)}, first 5 values: {vec[:5]}")