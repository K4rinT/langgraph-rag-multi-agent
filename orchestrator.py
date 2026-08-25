from __future__ import annotations
from functools import cache
from langgraph.graph import START, END, StateGraph
from agents import generate_node, retrieve_node
from state import RAGState

@cache
def build_workflow():
    """
    Build and compile the sequential RAG graph.
    The graph is stateless between runs, so it is complied once and reused across queries
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def run_query(question: str) -> RAGState:
    """
    Run one question through the graph and return final state
    """
    return build_workflow().invoke({"query": question})