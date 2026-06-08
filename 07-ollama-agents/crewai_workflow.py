"""
Multi-agent orchestration with CrewAI backed by a local Ollama model.

This example builds a 3-agent research & writing pipeline:
  1. Researcher  — gathers information on a topic
  2. Analyst     — synthesises findings into key insights
  3. Writer      — produces a polished final report

Prerequisites:
    pip install crewai crewai-tools ollama
    ollama pull llama3.2

Note: CrewAI uses LiteLLM under the hood. Prefix the model name with "ollama/"
      to route requests to your local Ollama instance.
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool  # optional — remove if you don't have a Serper key

# ── Model config ───────────────────────────────────────────────────────────────
# CrewAI routes "ollama/<model>" to http://localhost:11434 via LiteLLM
OLLAMA_MODEL = "ollama/llama3.2"

# Optional: web search tool (requires SERPER_API_KEY env var)
# Get a free key at https://serperdev.io — remove this block if not needed
search_tool = None
if os.getenv("SERPER_API_KEY"):
    search_tool = SerperDevTool()


# ── Agents ─────────────────────────────────────────────────────────────────────
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover comprehensive, accurate information about the assigned topic.",
    backstory=(
        "You are an experienced researcher with a talent for finding signal in noise. "
        "You synthesise information from multiple angles and always cite your reasoning."
    ),
    llm=OLLAMA_MODEL,
    tools=[search_tool] if search_tool else [],
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

analyst = Agent(
    role="Strategic Analyst",
    goal="Transform raw research into structured, actionable insights.",
    backstory=(
        "You excel at pattern recognition and turning complex information into clear, "
        "prioritised takeaways. You think in frameworks and bullet points."
    ),
    llm=OLLAMA_MODEL,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

writer = Agent(
    role="Technical Writer",
    goal="Produce a clear, well-structured report from the analyst's insights.",
    backstory=(
        "You write with precision and clarity. Your reports are concise, use plain English, "
        "and are structured so a busy executive can skim them in two minutes."
    ),
    llm=OLLAMA_MODEL,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)


# ── Task factory ───────────────────────────────────────────────────────────────
def build_tasks(topic: str) -> list[Task]:
    research_task = Task(
        description=(
            f"Research the following topic thoroughly:\n\n**{topic}**\n\n"
            "Cover: background & context, current state of the art, key players or technologies, "
            "recent developments, and open challenges. "
            "Produce detailed notes — do not summarise yet."
        ),
        expected_output=(
            "A detailed research brief of at least 400 words covering all the areas above, "
            "written in bullet points with sub-bullets where appropriate."
        ),
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            "Using the research brief, identify the 5 most important insights. "
            "For each insight explain: what it means, why it matters, and what action it implies. "
            "Focus on depth over breadth."
        ),
        expected_output=(
            "A structured list of exactly 5 insights, each with: "
            "Insight title, 2–3 sentence explanation, significance, and recommended action."
        ),
        agent=analyst,
        context=[research_task],
    )

    writing_task = Task(
        description=(
            "Write a polished 500-word executive report on the topic using the 5 insights. "
            "Structure: Executive Summary (2 sentences) → Key Findings (the 5 insights, formatted as H2 sections) "
            "→ Recommendations (3 bullet points) → Conclusion (1 paragraph). "
            "Tone: professional but accessible. No jargon."
        ),
        expected_output=(
            "A complete markdown-formatted report ready to be published or emailed. "
            "Must include all four sections listed above."
        ),
        agent=writer,
        context=[research_task, analysis_task],
    )

    return [research_task, analysis_task, writing_task]


# ── Crew builder ───────────────────────────────────────────────────────────────
def run_crew(topic: str) -> str:
    """
    Run the full 3-agent pipeline and return the final written report.

    Args:
        topic: The research topic or question.

    Returns:
        The writer agent's final markdown report as a string.
    """
    tasks = build_tasks(topic)

    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=tasks,
        process=Process.sequential,  # researcher → analyst → writer in order
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TOPIC = "The current state of local LLM inference — tools, models, and use cases in 2025"

    print(f"Starting CrewAI pipeline for topic:\n  '{TOPIC}'\n")
    print("=" * 60)

    report = run_crew(TOPIC)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report)

    # Save to disk
    output_path = "crew_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {TOPIC}\n\n{report}")
    print(f"\n✓ Report saved to {output_path}")