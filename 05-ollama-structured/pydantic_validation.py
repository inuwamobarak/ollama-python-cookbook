"""
Guaranteed structured output using Pydantic V2.

Ollama supports native structured output via the `format` parameter,
which forces the model to emit valid JSON matching schema.

Prerequisites:
    pip install ollama pydantic
    ollama pull llama3.2
"""

import json
from enum import Enum
from typing import Optional

import ollama
from pydantic import BaseModel, ValidationError


# ── Define schemas ────────────────────────────────────────────────────────
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewAnalysis(BaseModel):
    sentiment: Sentiment
    score: float          # 0.0 – 1.0 confidence
    summary: str          # one-sentence summary
    keywords: list[str]   # up to 5 keywords
    actionable: bool      # does the review suggest a concrete improvement?


class Person(BaseModel):
    name: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    location: Optional[str] = None


class CodeExplanation(BaseModel):
    language: str
    purpose: str
    complexity: str       # "beginner" | "intermediate" | "advanced"
    key_concepts: list[str]
    has_bugs: bool
    suggested_improvements: list[str]


# ── Generic extractor ─────────────────────────────────────────────────────────
def extract(prompt: str, schema: type[BaseModel], model: str = "llama3.2") -> BaseModel:
    """Send a prompt and parse the response into the given Pydantic model."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=schema.model_json_schema(),
    )
    raw = response["message"]["content"]

    try:
        return schema.model_validate_json(raw)
    except ValidationError as e:
        # Fallback: try stripping markdown fences if model added them
        print("Generic extractor Error: ", e)
        print("\nProcedding with fallback")
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return schema.model_validate(json.loads(clean))


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Sentiment analysis
    print("=== Review Analysis ===")
    review = (
        "The product arrived quickly but the packaging was damaged. "
        "The item itself works fine, though the instructions were confusing."
    )
    result: ReviewAnalysis = extract(
        f"Analyze this product review:\n\n{review}",
        ReviewAnalysis,
    )
    print(result.model_dump_json(indent=2))

    # 2. Entity extraction
    print("\n=== Person Extraction ===")
    bio = "Marie Curie was a 66-year-old Polish-French physicist and chemist who worked in Paris."
    person: Person = extract(
        f"Extract person information from this text:\n\n{bio}",
        Person,
    )
    print(person.model_dump_json(indent=2))

    # 3. Code analysis
    print("\n=== Code Explanation ===")
    code = """
def fib(n):
    if n <= 0: return []
    result = [0, 1]
    for i in range(2, n):
        result.append(result[-1] + result[-2])
    return result[:n]
"""
    explanation: CodeExplanation = extract(
        f"Analyze this code snippet:\n```\n{code}\n```",
        CodeExplanation,
    )
    print(explanation.model_dump_json(indent=2))