"""
Download a GGUF model from Hugging Face and register it with Ollama.

Prerequisites:
    pip install huggingface_hub ollama
"""

import shutil
from pathlib import Path

import ollama
from huggingface_hub import hf_hub_download


def download_gguf(
    repo_id: str,
    filename: str,
    cache_dir: Path = Path("./models"),
) -> Path:
    """Download a GGUF file from Hugging Face Hub and return its local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} from {repo_id} ...")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=cache_dir,
    )
    print(f"✓ Saved to {local_path}")
    return Path(local_path)


def register_with_ollama(
    gguf_path: Path,
    model_name: str,
    system_prompt: str = "You are a helpful assistant.",
) -> None:
    """Create an Ollama model entry pointing at a local GGUF file."""
    modelfile = f"""
FROM {gguf_path.resolve()}

PARAMETER temperature 0.7
PARAMETER num_ctx 4096

SYSTEM \"\"\"{system_prompt}\"\"\"
"""
    print(f"\nRegistering '{model_name}' with Ollama ...")
    for progress in ollama.create(model=model_name, modelfile=modelfile, stream=True):
        status = progress.get("status", "")
        if status:
            print(f"  {status}")

    print(f"✓ Model '{model_name}' registered. Run: ollama run {model_name}")


def cleanup(model_name: str, gguf_path: Path) -> None:
    """Remove the Ollama entry and the downloaded file."""
    ollama.delete(model_name)
    gguf_path.unlink(missing_ok=True)
    print(f"Cleaned up '{model_name}' and {gguf_path}")


if __name__ == "__main__":
    # Example: Qwen2.5-0.5B — tiny model for testing
    REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    MODEL_NAME = "qwen2.5-local"

    gguf_path = download_gguf(REPO_ID, FILENAME)
    register_with_ollama(gguf_path, MODEL_NAME)

    # Quick smoke test
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello in 5 different languages."}],
    )
    print("\n=== Response ===")
    print(response["message"]["content"])