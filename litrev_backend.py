"""
litrev_backend.py
=================
Groq-powered multi-agent literature review assistant
(Token-safe version)
"""

from __future__ import annotations
import os
import asyncio
from typing import AsyncGenerator, Dict, List

import arxiv
from dotenv import load_dotenv

from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ---------------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

# ---------------------------------------------------------------------------
# TOOL: arXiv search (TRUNCATED)
# ---------------------------------------------------------------------------

def arxiv_search(query: str, max_results: int = 5) -> List[Dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "published": result.published.strftime("%Y-%m-%d"),
            # 🔥 TRUNCATE ABSTRACT
            "summary": result.summary[:500] + "...",
            "pdf_url": result.pdf_url,
        })

    return papers


arxiv_tool = FunctionTool(
    arxiv_search,
    description="Search arXiv and return compact paper metadata."
)

# ---------------------------------------------------------------------------
# TEAM
# ---------------------------------------------------------------------------

def build_team() -> RoundRobinGroupChat:

    llm = OpenAIChatCompletionClient(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        model_info={
            "family": "llama",
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
        },
    )

    search_agent = AssistantAgent(
        name="search_agent",
        system_message=(
            "Search arXiv using the tool and return EXACTLY the number "
            "of papers requested. Keep output concise."
        ),
        tools=[arxiv_tool],
        model_client=llm,
        reflect_on_tool_use=False,  # 🔥 VERY IMPORTANT
    )

    summarizer = AssistantAgent(
        name="summarizer",
        system_message=(
            "Write a short literature review in Markdown:\n"
            "- 2 sentence intro\n"
            "- Bullet list of papers (title as link, authors, contribution)\n"
            "- 1 sentence conclusion"
        ),
        model_client=llm,
    )

    return RoundRobinGroupChat(
        participants=[search_agent, summarizer],
        max_turns=2,
    )

# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

async def run_litrev(
    topic: str,
    num_papers: int = 5,
) -> AsyncGenerator[str, None]:

    team = build_team()
    task = f"Conduct a literature review on {topic}. Return {num_papers} papers."

    async for msg in team.run_stream(task=task):
        if isinstance(msg, TextMessage):
            yield f"{msg.source}: {msg.content}"

# ---------------------------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def demo():
        async for line in run_litrev("Credit Card Fraud Detection", 5):
            print(line)

    asyncio.run(demo())
