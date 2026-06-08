"""
Agentic loops with Ollama's native function-calling API.

Ollama supports OpenAI-compatible tool use: define Python functions,
pass their schema to the model, execute called functions, and return results.

Prerequisites:
    pip install ollama
    ollama pull llama3.2   # ensure the model supports tool use
"""

import json
import math
from datetime import datetime

import ollama

MODEL = "llama3.2"


# ── Tool implementations ───────────────────────────────────────────────────────
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Safely evaluate a basic math expression."""
    try:
        # Restrict to safe math operations
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search_web(query: str) -> str:
    """Simulated web search (replace with a real API in production)."""
    return f"[Simulated search result for '{query}']: Top result would appear here."


def get_weather(city: str) -> str:
    """Simulated weather lookup (replace with a real API in production)."""
    return f"[Simulated] Weather in {city}: 22°C, partly cloudy."


# ── Tool registry: maps function name → callable ──────────────────────────────
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_web": search_web,
    "get_weather": get_weather,
}

# ── Tool schemas: passed to the model ─────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a mathematical expression. Supports Python math operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '2 ** 10' or 'sqrt(144)'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


# ── Agentic loop ───────────────────────────────────────────────────────────────
def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    Single-turn agentic loop:
    1. Send user message + tool schemas to the model.
    2. If the model requests tool calls, execute them.
    3. Return results to the model for a final response.
    Repeats until the model stops calling tools.
    """
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        msg = response["message"]

        # No tool calls → final answer
        if not msg.get("tool_calls"):
            return msg["content"]

        # Execute each requested tool call
        messages.append(msg)  # append assistant's tool-call request

        for tool_call in msg["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            fn_args = tool_call["function"]["arguments"]

            if verbose:
                print(f"  → Calling tool: {fn_name}({json.dumps(fn_args)})")

            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn is None:
                result = f"Unknown tool: {fn_name}"
            else:
                result = fn(**fn_args) if fn_args else fn()

            if verbose:
                print(f"  ← Result: {result}")

            messages.append({
                "role": "tool",
                "content": result,
            })


if __name__ == "__main__":
    queries = [
        "What time is it right now?",
        "What is the square root of 1764?",
        "What is 2 to the power of 32?",
        "What's the weather like in Tokyo?",
        "Search for the latest news about Ollama.",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        answer = run_agent(q)
        print(f"A: {answer}")