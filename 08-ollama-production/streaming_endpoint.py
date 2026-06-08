"""
FastAPI SSE streaming endpoint with a matching HTML test client.

Streams tokens from Ollama back to any frontend (browser, curl, JS EventSource).
Run this file directly to start the server; open http://localhost:8000 to test it
in a browser with the built-in chat UI.

Prerequisites:
    pip install fastapi uvicorn ollama
    ollama pull llama3.2

Run:
    python streaming_endpoint.py
    # then open http://localhost:8000
"""

import asyncio
import json
from typing import AsyncGenerator

import ollama
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Ollama Streaming API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class StreamRequest(BaseModel):
    message: str
    model: str = "llama3.2"
    system_prompt: str = "You are a helpful assistant."
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    history: list[dict] = []


# ── SSE helpers ────────────────────────────────────────────────────────────────
def sse_event(data: str, event: str = "token") -> str:
    """Format a single Server-Sent Event."""
    return f"event: {event}\ndata: {data}\n\n"


async def token_stream(request: StreamRequest) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted events:
      - event: token  — individual text tokens
      - event: done   — signals the stream is complete
      - event: error  — signals a generation error
    """
    messages = [
        {"role": "system", "content": request.system_prompt},
        *request.history,
        {"role": "user", "content": request.message},
    ]

    try:
        # Ollama's sync streaming works fine inside an async context for I/O-bound work;
        # wrap in run_in_executor for CPU-heavy models if needed.
        loop = asyncio.get_event_loop()

        def _sync_stream():
            return ollama.chat(
                model=request.model,
                messages=messages,
                options={"temperature": request.temperature},
                stream=True,
            )

        stream = await loop.run_in_executor(None, _sync_stream)

        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield sse_event(json.dumps({"token": token}))
            await asyncio.sleep(0)  # yield control to the event loop

        yield sse_event("{}", event="done")

    except ollama.ResponseError as exc:
        yield sse_event(json.dumps({"error": str(exc)}), event="error")
    except Exception as exc:
        yield sse_event(json.dumps({"error": f"Unexpected error: {exc}"}), event="error")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.post("/stream")
async def stream_chat(request: StreamRequest):
    """
    POST /stream — returns a text/event-stream response.

    Client-side usage (JavaScript):

        const source = new EventSource('/stream');  // for GET-based SSE
        // For POST, use fetch with ReadableStream:

        const resp = await fetch('/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: 'Hello!'})
        });
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            console.log(decoder.decode(value));
        }
    """
    return StreamingResponse(
        token_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.get("/models")
async def list_models():
    """GET /models — list available Ollama models."""
    models = ollama.list()
    return {"models": [m["name"] for m in models["models"]]}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Built-in test client (HTML) ────────────────────────────────────────────────
HTML_CLIENT = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ollama Streaming Test</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e8eaf0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 20px; }
  h1 { font-size: 1.4rem; margin-bottom: 24px; color: #4ade80; }
  #chat { width: 100%; max-width: 720px; display: flex; flex-direction: column; gap: 16px; }
  #messages { min-height: 320px; max-height: 60vh; overflow-y: auto; background: #1a1d24; border: 1px solid #252830; border-radius: 12px; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { line-height: 1.6; font-size: 0.9rem; }
  .msg.user { color: #60a5fa; }
  .msg.assistant { color: #e8eaf0; white-space: pre-wrap; }
  .msg.system { color: #636878; font-style: italic; font-size: 0.8rem; }
  #controls { display: flex; gap: 10px; }
  #input { flex: 1; padding: 12px 16px; background: #1a1d24; border: 1px solid #252830; border-radius: 10px; color: #e8eaf0; font-size: 0.9rem; outline: none; }
  #input:focus { border-color: #4ade80; }
  #send { padding: 12px 24px; background: #4ade80; color: #0f1117; border: none; border-radius: 10px; font-weight: 700; cursor: pointer; font-size: 0.9rem; }
  #send:disabled { opacity: 0.4; cursor: not-allowed; }
  #model-select { padding: 12px; background: #1a1d24; border: 1px solid #252830; border-radius: 10px; color: #e8eaf0; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>🦙 Ollama Streaming API — Test Client</h1>
<div id="chat">
  <div style="display:flex; gap:10px; align-items:center;">
    <label style="font-size:0.82rem; color:#636878;">Model:</label>
    <select id="model-select"><option>llama3.2</option></select>
  </div>
  <div id="messages">
    <div class="msg system">Connected to streaming endpoint. Send a message to begin.</div>
  </div>
  <div id="controls">
    <input id="input" type="text" placeholder="Type a message and press Enter..." autofocus>
    <button id="send">Send</button>
  </div>
</div>

<script>
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const modelSelect = document.getElementById('model-select');
const history = [];

// Load available models
fetch('/models').then(r => r.json()).then(data => {
  modelSelect.innerHTML = data.models.map(m => `<option>${m}</option>`).join('');
});

function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = role === 'user' ? `You: ${content}` : content;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

async function send() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = '';
  sendBtn.disabled = true;

  addMessage('user', text);
  history.push({ role: 'user', content: text });

  const assistantDiv = addMessage('assistant', '');
  let fullResponse = '';

  try {
    const resp = await fetch('/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        model: modelSelect.value,
        history: history.slice(0, -1),  // exclude the message we just added
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse SSE lines
      const lines = buffer.split('\\n');
      buffer = lines.pop();  // keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const payload = JSON.parse(line.slice(6));
            if (payload.token) {
              fullResponse += payload.token;
              assistantDiv.textContent = fullResponse;
              messagesEl.scrollTop = messagesEl.scrollHeight;
            }
          } catch {}
        }
      }
    }

    history.push({ role: 'assistant', content: fullResponse });
  } catch (err) {
    assistantDiv.textContent = `Error: ${err.message}`;
    assistantDiv.style.color = '#f87171';
  }

  sendBtn.disabled = false;
  inputEl.focus();
}

sendBtn.addEventListener('click', send);
inputEl.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def test_client():
    """Serve the built-in browser test UI at http://localhost:8000."""
    return HTML_CLIENT


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Ollama Streaming API ...")
    print("  POST /stream  — SSE streaming endpoint")
    print("  GET  /models  — list available models")
    print("  GET  /health  — health check")
    print("  GET  /        — browser test client")
    print()
    print("Open http://localhost:8000 in your browser to test.")
    uvicorn.run("streaming_endpoint:app", host="0.0.0.0", port=8000, reload=True)