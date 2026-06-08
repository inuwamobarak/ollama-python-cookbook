"""
Production-grade FastAPI wrapper around Ollama.

Features:
  • /chat         — standard JSON chat endpoint
  • /stream       — streaming SSE endpoint
  • /models       — list available models
  • /health       — health check
  • API key auth  — configurable via environment variable
  • CORS          — configurable origins
  • Rate limiting — via slowapi

Prerequisites:
    pip install fastapi uvicorn ollama slowapi python-dotenv
    ollama pull llama3.2

Run:
    uvicorn fastapi_server:app --reload --port 8000

Test:
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -H "X-API-Key: dev-secret" \
         -d '{"message": "What is Ollama?", "model": "llama3.2"}'
"""

import os
from typing import AsyncGenerator

import ollama
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "dev-secret")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── App setup ──────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Ollama API", version="1.0.0", description="Local LLM inference via Ollama")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── Auth ───────────────────────────────────────────────────────────────────────
async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
    return api_key


# ── Schemas ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    model: str = DEFAULT_MODEL
    system_prompt: str = "You are a helpful assistant."
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    history: list[dict] = []


class ChatResponse(BaseModel):
    response: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check — also verifies Ollama is reachable."""
    try:
        models = ollama.list()
        return {"status": "ok", "models_available": len(models["models"])}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {e}")


@app.get("/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """List all locally available Ollama models."""
    models = ollama.list()
    return {"models": [m["name"] for m in models["models"]]}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def chat(request: ChatRequest):
    """Standard blocking chat endpoint."""
    messages = [
        {"role": "system", "content": request.system_prompt},
        *request.history,
        {"role": "user", "content": request.message},
    ]

    try:
        response = ollama.chat(
            model=request.model,
            messages=messages,
            options={"temperature": request.temperature},
        )
    except ollama.ResponseError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return ChatResponse(
        response=response["message"]["content"],
        model=request.model,
    )


@app.post("/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def stream_chat(request: ChatRequest):
    """Streaming SSE endpoint — tokens arrive as they are generated."""
    messages = [
        {"role": "system", "content": request.system_prompt},
        *request.history,
        {"role": "user", "content": request.message},
    ]

    async def token_generator() -> AsyncGenerator[str, None]:
        try:
            for chunk in ollama.chat(
                model=request.model,
                messages=messages,
                options={"temperature": request.temperature},
                stream=True,
            ):
                token = chunk["message"]["content"]
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except ollama.ResponseError as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=True)