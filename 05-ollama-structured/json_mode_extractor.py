"""
Structured JSON extraction using Ollama's raw format="json" mode.

Unlike the Pydantic approach, this uses the lower-level format flag and parses
the output manually — useful when you want flexibility or don't want Pydantic.

Prerequisites:
    pip install ollama
    ollama pull llama3.2
"""

import json
from typing import Any

import ollama

MODEL = "llama3.2"


# ── Core extractor ────────────────────────────────────────────────────────────
def extract_json(
    prompt: str,
    schema_hint: str = "",
    model: str = MODEL,
) -> dict[str, Any]:
    """
    Ask the model to respond in JSON and parse the result.

    Args:
        prompt:      The user instruction.
        schema_hint: Optional description of expected fields, injected into system prompt.
        model:       Ollama model name.

    Returns:
        Parsed Python dict.

    Raises:
        ValueError: If the response cannot be decoded as JSON.
    """
    system = "You are a data extractor. Respond ONLY with valid JSON. No markdown, no explanation."
    if schema_hint:
        system += f"\n\nExpected JSON structure:\n{schema_hint}"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        format="json",
    )

    raw: str = response["message"]["content"]

    # Defensive: strip accidental markdown fences some models still emit
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON:\n{raw}") from exc


# ── Convenience wrappers ──────────────────────────────────────────────────────
def extract_contact(text: str) -> dict:
    return extract_json(
        prompt=f"Extract contact information from:\n\n{text}",
        schema_hint='{"name": str, "email": str | null, "phone": str | null, "company": str | null}',
    )


def extract_event(text: str) -> dict:
    return extract_json(
        prompt=f"Extract event details from:\n\n{text}",
        schema_hint='{"title": str, "date": str | null, "time": str | null, "location": str | null, "description": str}',
    )


def extract_product(text: str) -> dict:
    return extract_json(
        prompt=f"Extract product information from this listing:\n\n{text}",
        schema_hint='{"name": str, "price": float | null, "currency": str, "features": list[str], "in_stock": bool}',
    )


def batch_extract(items: list[str], extractor_fn) -> list[dict]:
    """Run the same extractor over a list of texts."""
    results = []
    for i, item in enumerate(items, 1):
        print(f"  Extracting {i}/{len(items)} ...", end="\r")
        results.append(extractor_fn(item))
    print()
    return results


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Contact extraction ===")
    contact_text = (
        "Please reach out to Dr. Sarah Chen at sarah.chen@acmecorp.io "
        "or call her on +1-415-555-0192. She leads the AI division at Acme Corp."
    )
    contact = extract_contact(contact_text)
    print(json.dumps(contact, indent=2))

    print("\n=== Event extraction ===")
    event_text = (
        "Join us for the Python Meetup on Thursday, 19 June 2025 at 7:00 PM. "
        "We'll be at the TechHub Berlin, Rosenthaler Str. 40. Talk: Async Python patterns."
    )
    event = extract_event(event_text)
    print(json.dumps(event, indent=2))

    print("\n=== Product extraction ===")
    product_text = (
        "Keychron K2 Pro — $99.99. Hot-swappable, QMK/VIA compatible, "
        "Bluetooth 5.1, USB-C. Available in aluminum frame. Ships in 3–5 days."
    )
    product = extract_product(product_text)
    print(json.dumps(product, indent=2))

    print("\n=== Batch contact extraction ===")
    contacts = batch_extract(
        [
            "Email John Doe at john@example.com, CTO of StartupX.",
            "Reach Maria García via maria.garcia@techco.es — Senior Engineer.",
            "Call Bob Smith on 020-7946-0123, works at FinanceCo Ltd.",
        ],
        extract_contact,
    )
    for c in contacts:
        print(json.dumps(c, indent=2))