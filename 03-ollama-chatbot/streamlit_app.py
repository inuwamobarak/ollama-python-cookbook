"""
Browser chat interface backed by a local Ollama model.

Prerequisites:
    pip install streamlit ollama
    ollama pull llama3.2

Run:
    streamlit run streamlit_app.py
"""

import ollama
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ollama Chat", page_icon="🦙", layout="centered")
st.title("🦙 Ollama Chat")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    available_models = [m["name"] for m in ollama.list()["models"]]
    model = st.selectbox("Model", available_models, index=0)

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, concise assistant.",
        height=120,
    )

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)

    if st.button("🗑️ Clear history"):
        st.session_state.messages = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle new user input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages with system prompt prepended
    messages_with_system = [
        {"role": "system", "content": system_prompt},
        *st.session_state.messages,
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in ollama.chat(
            model=model,
            messages=messages_with_system,
            options={"temperature": temperature},
            stream=True,
        ):
            full_response += chunk["message"]["content"]
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})