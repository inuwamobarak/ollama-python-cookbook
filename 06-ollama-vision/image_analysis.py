"""
Multi-modal image understanding with LLaVA / llama3.2-vision.

Prerequisites:
    pip install ollama pillow requests
    ollama pull llama3.2-vision   # or: llava, moondream
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Union

import ollama
import requests
from PIL import Image

VISION_MODEL = "llama3.2-vision"  # swap for "llava" or "moondream"


# ── Image loading helpers ──────────────────────────────────────────────────────
def load_image_from_path(path: Union[str, Path]) -> bytes:
    return Path(path).read_bytes()


def load_image_from_url(url: str) -> bytes:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.content


def resize_for_model(image_bytes: bytes, max_pixels: int = 1024) -> bytes:
    """Optionally downscale large images to keep inference fast."""
    img = Image.open(BytesIO(image_bytes))
    w, h = img.size
    if max(w, h) > max_pixels:
        ratio = max_pixels / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ── Core analysis function ────────────────────────────────────────────────────
def analyze_image(
    image_bytes: bytes,
    prompt: str = "Describe this image in detail.",
    model: str = VISION_MODEL,
) -> str:
    """Send image bytes + prompt to an Ollama vision model."""
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_bytes],
            }
        ],
    )
    return response["message"]["content"]


def analyze_image_streaming(
    image_bytes: bytes,
    prompt: str,
    model: str = VISION_MODEL,
) -> str:
    """Stream the analysis response token-by-token."""
    full = ""
    for chunk in ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
        stream=True,
    ):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full += token
    print()
    return full


# ── Multi-turn visual conversation ───────────────────────────────────────────
def visual_chat_loop(image_bytes: bytes, model: str = VISION_MODEL) -> None:
    """Hold a multi-turn conversation about an image."""
    history = [{"role": "user", "content": "Here is the image I want to discuss.", "images": [image_bytes]}]

    # First turn: auto-describe
    response = ollama.chat(model=model, messages=history)
    initial_desc = response["message"]["content"]
    history.append({"role": "assistant", "content": initial_desc})
    print(f"Model: {initial_desc}\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        history.append({"role": "user", "content": user_input})
        response = ollama.chat(model=model, messages=history)
        reply = response["message"]["content"]
        history.append({"role": "assistant", "content": reply})
        print(f"Model: {reply}\n")


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Download a sample image
    SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    print("Downloading sample image ...")
    img_bytes = load_image_from_url(SAMPLE_URL)
    img_bytes = resize_for_model(img_bytes)

    print("\n=== Basic description ===")
    print(analyze_image(img_bytes, "What do you see in this image?"))

    print("\n=== Streaming analysis ===")
    analyze_image_streaming(img_bytes, "List all the objects you can identify.")

    # Uncomment to load a local image:
    # img_bytes = load_image_from_path("./my_image.jpg")
    # print(analyze_image(img_bytes, "What text appears in this image?"))