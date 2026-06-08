"""
Full RAG pipeline: ingest → embed → store → retrieve → generate.

Prerequisites:
    pip install ollama chromadb pypdf langchain-text-splitters
    ollama pull llama3.2 nomic-embed-text
"""

from pathlib import Path

import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 3


# ── Clients ────────────────────────────────────────────────────────────────────
chroma_client = chromadb.Client()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def get_or_create_collection() -> chromadb.Collection:
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# ── Ingest ─────────────────────────────────────────────────────────────────────
def ingest_text(text: str, doc_id: str = "doc") -> int:
    """Split text into chunks, embed them, and store in ChromaDB."""
    collection = get_or_create_collection()
    chunks = splitter.split_text(text)

    embeddings = [
        ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
        for chunk in chunks
    ]

    ids = [f"{doc_id}-chunk-{i}" for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)

    print(f"✓ Ingested {len(chunks)} chunks from '{doc_id}'")
    return len(chunks)


def ingest_file(path: Path) -> int:
    """Read a .txt or .md file and ingest it."""
    text = path.read_text(encoding="utf-8")
    return ingest_text(text, doc_id=path.stem)


# ── Retrieve ───────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> list[str]:
    """Embed the query and return the top-k most relevant chunks."""
    collection = get_or_create_collection()
    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results["documents"][0]


# ── Generate ───────────────────────────────────────────────────────────────────
def rag_query(question: str) -> str:
    """Retrieve relevant context and generate a grounded answer."""
    context_chunks = retrieve(question)
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ── Demo ───────────────────────────────────────────────────────────────────────
SAMPLE_TEXT = """
Ollama is a tool that allows you to run large language models locally on your machine.
It supports models like Llama 3, Mistral, Phi-3, Gemma 2, and many others.
Ollama provides a REST API on port 11434 by default.
The official Python SDK is called ollama-python and can be installed via pip.
Ollama uses Modelfiles to configure model behaviour, similar to Dockerfiles.
Models can be downloaded with the 'ollama pull' command.
Custom models can be created using 'ollama create' with a Modelfile.
Ollama supports GPU acceleration on NVIDIA, AMD, and Apple Silicon hardware.
The nomic-embed-text model is commonly used for generating text embeddings with Ollama.
ChromaDB is an open-source vector database that runs entirely in-process.
"""

if __name__ == "__main__":
    print("=== Ingesting sample document ===")
    ingest_text(SAMPLE_TEXT, doc_id="ollama-overview")

    questions = [
        "What port does Ollama use by default?",
        "How do I install the Python SDK?",
        "Which hardware does Ollama support for GPU acceleration?",
        "What is the capital of France?",  # Should return "I don't know"
    ]

    print("\n=== RAG Q&A ===")
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {rag_query(q)}")