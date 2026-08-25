from __future__ import annotations
from functools import cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent

@cache
def load_prompt(name: str) -> str:
    """
    Read a prompt by file stem
    """
    path = PROMPT_DIR / f"{name}.md"

    if not path.exists():
        available = sorted(p.stem for p in PROMPT_DIR.glob("*.md"))
        raise FileNotFoundError(
            f"No prompt named {name!r} in {PROMPT_DIR}. Available: {available}"
        )
    return path.read_text(encoding="utf-8").strip()

RETRIEVER_SYSTEM_PROMPT = load_prompt("retriever")
REPORT_GENERATOR_SYSTEM_PROMPT = load_prompt("report_generator")
NO_RESULT_MESSAGE = load_prompt("no_results")

__all__ = [
    "RETRIEVER_SYSTEM_PROMPT",
    "REPORT_GENERATOR_SYSTEM_PROMPT",
    "NO_RESULT_MESSAGE",
    "load_prompt"
]
