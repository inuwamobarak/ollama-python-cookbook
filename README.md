<div align="center">

<img src="https://img.shields.io/github/stars/inuwamobarak/ollama-python-cookbook?style=for-the-badge&color=FFD700" alt="Stars">
<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Ollama-Compatible-00B4D8?style=for-the-badge" alt="Ollama">
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT">

# 🦙 Ollama Python Cookbook

**The most complete collection of starter to production-ready Python patterns for local LLMs.**


[Quick Start](#-quick-start) • [What's Inside](#-whats-inside) • [Recipes](#-recipes) • [Contributing](#-contributing)

</div>

---

## Why this exists

You want to use Ollama for your project but not sure where to start? Every Ollama tutorial shows you `ollama.chat()`. Here we show you how to build a **streaming FastAPI backend**, wire up a **ChromaDB RAG pipeline**, extract **structured Pydantic models**, or run **parallel async inference**.

---

## What is Ollama?

Simple put, Ollama is a tool that helps use use and manage large language models (LLMs). It is arguably the easiest way to get up and running with large language models such as gpt-oss, Gemma 4, DeepSeek-R1, Qwen3 and more. If you are considering using LLMs in your workload then Ollama is a place to look at.

---

## ⚡ Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/inuwamobarak/ollama-python-cookbook
cd ollama-python-cookbook
```
```bash
# 2. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```
![alt text](<img/Screenshot 2026-06-08 at 20.45.33-2.png>)
```bash
# 3. Install deps (or per-section - see each folder)
python -m pip install ollama
```
```bash
# 4. Pull a model
ollama pull llama3.2
```
![alt text](<img/Screenshot 2026-06-08 at 20.52.39-1.png>)
```bash
# 5. Run your first script
python 01-ollama-inference/sync_client.py
```
![alt text](<img/Screenshot 2026-06-08 at 21.06.50-1.png>)

For Windows: https://ollama.com/download/windows

You might also want to use a separate env for this project. Use which ever you are familiar with. conda or python venv.

NOTE: Installing the Ollama application via curl downloads the system-level application, but it does not automatically install the Python library required for the scripts. Hence, you might need python -m pip install ollama

---

## 📦 What's Inside

```
ollama-python-cookbook/
├── 01-ollama-inference/      # SDK fundamentals & async patterns
├── 02-ollama-custom-models/  # Modelfiles & GGUF downloads
├── 03-ollama-chatbot/        # CLI & Streamlit chat UIs
├── 04-ollama-rag/            # Full RAG pipeline with ChromaDB
├── 05-ollama-structured/     # Pydantic V2 structured outputs
├── 06-ollama-vision/         # Multi-modal with LLaVA/llama3.2-vision
├── 07-ollama-agents/         # Native tool calling & agentic loops
└── 08-ollama-production/     # FastAPI server with streaming & auth
```

---

## 🍳 Recipes

### 01 · Inference Basics

| Script | What it demonstrates |
|--------|----------------------|
| [`sync_client.py`](01-ollama-inference/sync_client.py) | `ollama.generate()`, `ollama.chat()`, list models |
| [`streaming_client.py`](01-ollama-inference/streaming_client.py) | Real-time token streaming to terminal |
| [`async_inference.py`](01-ollama-inference/async_inference.py) | `AsyncClient` + `asyncio.gather` for parallel calls |
| [`openai_sdk_compat.py`](01-ollama-inference/openai_sdk_compat.py) | Drop-in OpenAI SDK replacement pointing at Ollama |

**Highlight — parallel async inference:**

```python
import asyncio, ollama

async def ask(prompt):
    client = ollama.AsyncClient()
    r = await client.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"]

# Run 4 prompts at the same time
results = asyncio.run(asyncio.gather(*[ask(p) for p in prompts]))
```

---

### 02 · Custom Models

| Script | What it demonstrates |
|--------|----------------------|
| [`create_custom_model.py`](02-ollama-custom-models/create_custom_model.py) | Build a Modelfile in Python, call `ollama.create()` |
| [`hf_gguf_downloader.py`](02-ollama-custom-models/hf_gguf_downloader.py) | Download any GGUF from Hugging Face Hub, auto-register |

**Highlight — custom persona in 5 lines:**

```python
import ollama

MODELFILE = """
FROM llama3.2
SYSTEM "You are a senior Python engineer. Always use type hints."
PARAMETER temperature 0.3
"""
ollama.create(model="python-expert", modelfile=MODELFILE)
```

---

### 03 · Chatbot UIs

| Script | What it demonstrates |
|--------|----------------------|
| [`cli_memory_chat.py`](03-ollama-chatbot/cli_memory_chat.py) | Terminal chat with full conversation memory |
| [`streamlit_app.py`](03-ollama-chatbot/streamlit_app.py) | Browser UI — model picker, system prompt, streaming |

**Run the Streamlit app:**

```bash
pip install streamlit ollama
streamlit run 03-ollama-chatbot/streamlit_app.py
```

---

### 04 · RAG Pipeline

| Script | What it demonstrates |
|--------|----------------------|
| [`chroma_pipeline.py`](04-ollama-rag/chroma_pipeline.py) | End-to-end: chunk → embed → store → retrieve → generate |

**Full pipeline in ~20 lines:**

```python
# 1. Chunk & embed documents
chunks = splitter.split_text(document)
embeddings = [ollama.embeddings(model="nomic-embed-text", prompt=c)["embedding"] for c in chunks]
collection.add(ids=[...], embeddings=embeddings, documents=chunks)

# 2. Query
query_vec = ollama.embeddings(model="nomic-embed-text", prompt=question)["embedding"]
results = collection.query(query_embeddings=[query_vec], n_results=3)

# 3. Generate grounded answer
context = "\n".join(results["documents"][0])
response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": f"Context:\n{context}\n\nQ: {question}"}])
```

**Prerequisites:** `pip install chromadb langchain-text-splitters`

---

### 05 · Structured Outputs

| Script | What it demonstrates |
|--------|----------------------|
| [`pydantic_validation.py`](05-ollama-structured/pydantic_validation.py) | Schema-enforced JSON with Pydantic V2 |
| [`json_mode_extractor.py`](05-ollama-structured/json_mode_extractor.py) | Raw `format="json"` + `json.loads` fallback |

**Highlight — guaranteed structure:**

```python
from pydantic import BaseModel
import ollama

class ReviewAnalysis(BaseModel):
    sentiment: str
    score: float
    keywords: list[str]

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": f"Analyze: {review}"}],
    format=ReviewAnalysis.model_json_schema(),   # 🔑 forces valid JSON
)
result = ReviewAnalysis.model_validate_json(response["message"]["content"])
print(result.sentiment)  # always a valid ReviewAnalysis object
```

---

### 06 · Vision

| Script | What it demonstrates |
|--------|----------------------|
| [`image_analysis.py`](06-ollama-vision/image_analysis.py) | Describe images, multi-turn visual chat |
| [`video_frame_ocr.py`](06-ollama-vision/video_frame_ocr.py) | Extract frames from video, OCR with Ollama |

```bash
ollama pull llama3.2-vision   # or: llava, moondream
python 06-ollama-vision/image_analysis.py
```

---

### 07 · Agents & Tool Calling

| Script | What it demonstrates |
|--------|----------------------|
| [`native_tool_calling.py`](07-ollama-agents/native_tool_calling.py) | Full agentic loop with Python functions as tools |
| [`crewai_workflow.py`](07-ollama-agents/crewai_workflow.py) | Multi-agent orchestration with CrewAI |

**Highlight — tool calling loop:**

```python
# 1. Define tools as Python functions + JSON schemas
# 2. Send to model
response = ollama.chat(model="llama3.2", messages=messages, tools=TOOLS)

# 3. Execute what the model requested
for tool_call in response["message"]["tool_calls"]:
    result = my_functions[tool_call["function"]["name"]](**tool_call["function"]["arguments"])
    messages.append({"role": "tool", "content": result})

# 4. Send results back for final answer
final = ollama.chat(model="llama3.2", messages=messages)
```

---

### 08 · Production API

| Script | What it demonstrates |
|--------|----------------------|
| [`fastapi_server.py`](08-ollama-production/fastapi_server.py) | REST API with auth, rate limiting, CORS |
| [`streaming_endpoint.py`](08-ollama-production/streaming_endpoint.py) | SSE streaming to any frontend |

```bash
pip install fastapi uvicorn slowapi python-dotenv
uvicorn 08-ollama-production.fastapi_server:app --reload

# Chat
curl -X POST http://localhost:8000/chat \
     -H "X-API-Key: dev-secret" \
     -H "Content-Type: application/json" \
     -d '{"message": "Explain RAG in one paragraph."}'

# Stream
curl -N http://localhost:8000/stream \
     -X POST \
     -H "X-API-Key: dev-secret" \
     -H "Content-Type: application/json" \
     -d '{"message": "Write a poem about local AI."}'
```

---

## 🔧 Model Recommendations

| Task | Recommended Model | Pull Command |
|------|-------------------|--------------|
| General chat | `llama3.2` | `ollama pull llama3.2` |
| Code generation | `qwen2.5-coder` | `ollama pull qwen2.5-coder` |
| Embeddings | `nomic-embed-text` | `ollama pull nomic-embed-text` |
| Vision | `llama3.2-vision` | `ollama pull llama3.2-vision` |
| Fast / lightweight | `phi3.5` | `ollama pull phi3.5` |
| Structured output | `mistral` | `ollama pull mistral` |

---

## 🤝 Contributing

PRs are welcome! To add a new recipe:

1. Fork the repo
2. Create your script in the relevant `0X-` folder
3. Follow the template: module docstring with prerequisites, clean typed code, `if __name__ == "__main__"` demo block
4. Open a PR with a one-line description

---

## 📄 License

MIT — use freely in personal and commercial projects.

---

<div align="center">

**Found this useful? Give it a ⭐ — it helps others discover the cookbook.**

Built with 🦙 by the community [`Our LinkedIn Group`](https://www.linkedin.com/groups/14432232/) • Not affiliated with Ollama

</div>