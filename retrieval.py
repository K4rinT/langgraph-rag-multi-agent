from __future__ import annotations
import json
import logging
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from langchain_core.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.txt"
TOP_K = 5
MIN_SCORE = 0.05


@dataclass(frozen=True)
class Snippet:
    """
    Retrieve passage, with the score that earned it
    """
    chunk_id: int
    score: float
    text: str

def _is_heading(block: str) -> bool:
    """
    A lone short line with no closing full stop is section title.
    """
    return (
        "\n" not in block
        and len(block) < 100
        and not block.endswith(".")
    )


def load_chunks(path: Path = KNOWLEDGE_BASE_PATH) -> list[str]:
    """
    Split the knowledge_base into paragraph chunks, with heading
    """
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {path}")
    raw = path.read_text(encoding="utf-8")

    blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]

    chunks: list[str] = []
    heading = ""
    index = 0
    while index < len(blocks):
        block = blocks[index]
        followed_by_prose = index + 1 < len(blocks) and not _is_heading(blocks[index + 1])
        if _is_heading(block) and followed_by_prose:
            heading = block
            index += 1
            continue
        chunks.append(f"{heading}\n{block}" if heading else block)
        index +=1

    return chunks


@cache
def _index():
    """
    Build tf-idf index and reuse it across calls
    """
    chunks = load_chunks()
    if not chunks:
        raise ValueError(
            f"Knowledge base at {KNOWLEDGE_BASE_PATH} has no content. "
            "Add paragraphs separated by blank lines."
        )
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks)
    logger.info(f"Indexed {len(chunks)} chunks from {KNOWLEDGE_BASE_PATH.name} ({matrix.shape[1]} features)")

    return chunks, vectorizer, matrix


def warm_index() -> None:
    """
    Build index up front, at startup
    """
    _index()

def search(query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> list[Snippet]:
    """
    Return highest scoring chunks for query
    """
    chunks, vectorizer, matrix = _index()
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    results = [
        Snippet(chunk_id = i, score = round(float(score), 4), text = chunks[i])
        for i, score in ranked[:top_k]
        if score >= min_score
    ]

    if results:
        logger.debug(f"search({query!r}) -> {[(s.chunk_id, s.score) for s in results]}")
    else:
        best = round(float(ranked[0][1]), 4) if ranked else 0.0
        logger.debug(f"search({query!r}) -> no hits above {min_score:.2f} (best score {best:.4f})")

    return results


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search internal knowledge_base for passage relevant to query
    """
    results = search(query)

    return json.dumps([asdict(snippet) for snippet in results], ensure_ascii=False)