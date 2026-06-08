"""
Run multiple inference calls concurrently with asyncio.

Prerequisites:
    pip install ollama
    ollama pull llama3.2
"""

import asyncio
import time
import ollama


async def async_chat(prompt: str, model: str = "llama3.2") -> dict:
    """Single async chat call."""
    client = ollama.AsyncClient()
    start = time.perf_counter()
    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - start
    return {
        "prompt": prompt,
        "response": response["message"]["content"],
        "elapsed_s": round(elapsed, 2),
    }


async def run_parallel(prompts: list[str], model: str = "llama3.2") -> list[dict]:
    """Execute all prompts concurrently and return results in order."""
    tasks = [async_chat(p, model) for p in prompts]
    return await asyncio.gather(*tasks)


async def stream_async(prompt: str, model: str = "llama3.2") -> str:
    """Async streaming example."""
    client = ollama.AsyncClient()
    full = ""
    async for chunk in await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full += token
    print()
    return full


if __name__ == "__main__":
    prompts = [
        "What is 2 + 2?",
        "Name the largest planet in the solar system.",
        "Who wrote Romeo and Juliet?",
        "What language is Ollama written in?",
    ]

    print("=== Parallel inference (4 prompts concurrently) ===")
    wall_start = time.perf_counter()
    results = asyncio.run(run_parallel(prompts))
    wall_elapsed = time.perf_counter() - wall_start

    for r in results:
        print(f"Q: {r['prompt']}")
        print(f"A: {r['response'].strip()}")
        print(f"   [{r['elapsed_s']}s]\n")

    print(f"Total wall time: {wall_elapsed:.2f}s (vs ~{sum(r['elapsed_s'] for r in results):.2f}s sequential)")

    print("\n=== Async streaming ===")
    asyncio.run(stream_async("Explain async/await in one paragraph."))