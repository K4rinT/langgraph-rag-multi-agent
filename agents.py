from __future__ import annotations
import json
import os
import time
import logging

from functools import cache
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from llm import build_llm
from prompts import (
    NO_RESULT_MESSAGE,
    RETRIEVER_SYSTEM_PROMPT,
    REPORT_GENERATOR_SYSTEM_PROMPT
)
from retrieval import Snippet, search_knowledge_base
from state import RAGState

logger = logging.getLogger(__name__)

MAX_SNIPPETS = 8

@cache
def _retriever_agent():
    """
    Build the Data Retriever agent on first use
    """
    effort = os.getenv("RETRIEVER_REASONING_EFFORT", "minimal")
    options = {"reasoning_effort": effort} if effort else {}
    return create_react_agent(
        model=build_llm(**options),
        tools=[search_knowledge_base],
        prompt=RETRIEVER_SYSTEM_PROMPT
    )

@cache
def _report_generator_llm():
    """
    Build the Report Generator client on first use
    """
    return build_llm()


def _collect_snippets(messages) -> list[Snippet]:
    """
    Pull retrieved passages out of the agent's tool messages.
    """
    seen: dict[int, Snippet] = {}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            try:
                snippet = Snippet(**item)
            except TypeError:
                logger.debug(f"Ignoring malformed snippet payload: {item!r}")
                continue
            existing = seen.get(snippet.chunk_id)
            if existing is None or snippet.score > existing.score:
                seen[snippet.chunk_id] = snippet

    return sorted(seen.values(), key=lambda s: s.score, reverse=True)


def _extract_search_terms(messages) -> list[str]:
    """
    Phrases, agent chose to search for
    """
    terms: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            query = (call.get("args") or {}).get("query")
            if query:
                terms.append(query)

    return terms


def retrieve_node(state: RAGState) -> RAGState:
    """
    Graph node: Run Data Retriever and return raw snippets
    """
    logger.info(f"Retrieving for query: {state['query']}")
    started = time.perf_counter()

    result = _retriever_agent().invoke(
        {"messages": [HumanMessage(content=state["query"])]}
    )

    search_terms = _extract_search_terms(result["messages"])

    logger.info(f"Retriever searched {search_terms}")
    merged = _collect_snippets(result["messages"])
    snippets = merged[:MAX_SNIPPETS]
    if len(merged) > len(snippets):
        logger.debug(f"Trimmed {len(merged)} merged snippet(s) to the top {MAX_SNIPPETS}")
    logger.info(f"Retrieved {len(snippets)} unique snippet(s) in {time.perf_counter() - started:.2f}s: {[s.chunk_id for s in snippets]}")

    if not snippets:
        logger.warning("No snippets found; generator will return the no-result message")

    return {"search_terms": search_terms, "snippets": snippets}


def _format_reference_material(snippets: list[Snippet]) -> str:
    return "\n\n".join(
        f"[Excerpt {i}]\n{snippet.text}" for i, snippet in enumerate(snippets, start=1)
    )


def generate_node(state: RAGState) -> RAGState:
    """
    Graph node: run the Report Generator over the retrieved snippets
    """
    snippets = state.get("snippets", [])
    if not snippets:
        return {"answer": NO_RESULT_MESSAGE}

    user_message = (
        f"Question:\n{state['query']}\n\n"
        f"Reference material:\n{_format_reference_material(snippets)}"
    )
    logger.debug(f"Generator input ({len(user_message)} chars):\n{user_message}")
    started = time.perf_counter()

    response = _report_generator_llm().invoke(
        [
            SystemMessage(content=REPORT_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]
    )

    usage = getattr(response, "usage_metadata", None) or {}
    logger.info(
        f"Generated answer in {time.perf_counter() - started:.2f}s "
        f"({usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out tokens)"
    )

    return {"answer": response.text}