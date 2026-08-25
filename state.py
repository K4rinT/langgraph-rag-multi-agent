from __future__ import annotations
from typing import TypedDict
from retrieval import Snippet

class RAGState(TypedDict, total=False):
    """
    Schema for the state LangGraph threads between agents.
    
    Caller
        Writes
            1. query
    Agent 1: Data Retriever 
        Reads
            1. query
        Writes
            1. search_terms
            2. snippets
    Agent 2: Report Generator 
        Reads
            1. query
            2. snippets
        Writes
            1. answer
    """
    query: str
    search_terms: list[str]
    snippets: list[Snippet]
    answer: str