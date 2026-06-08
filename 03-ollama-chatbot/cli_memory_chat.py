"""
Terminal chatbot with persistent conversation memory.

Prerequisites:
    pip install ollama
    ollama pull llama3.2
"""

import ollama

MODEL = "llama3.2"
SYSTEM_PROMPT = "You are a concise, friendly assistant. Keep replies under 3 sentences unless asked for more."


def chat_loop() -> None:
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatting with {MODEL}. Type 'quit' to exit, 'reset' to clear history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[History cleared]\n")
            continue

        history.append({"role": "user", "content": user_input})

        # Stream the response token-by-token
        print(f"{MODEL}: ", end="", flush=True)
        full_response = ""

        for chunk in ollama.chat(model=MODEL, messages=history, stream=True):
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token

        print()  # newline after response

        # Append the assistant reply to maintain context
        history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    chat_loop()