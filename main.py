from __future__ import annotations
import argparse
import logging
from dotenv import load_dotenv
from orchestrator import run_query
from retrieval import warm_index
from log_config import setup_logging

logger = logging.getLogger(__name__)

SAMPLE_QUERIES = [
    "What is the policy on international travel?",
    "I'm taking a laptop to a restricted country next month -- what do I need to do?",
    "How many days of leave do I get?",
    "Do I have to pay back training costs if I quit?",
    "What is the parental leave policy?"
]

def show(question: str) -> None:
    """
    Run one question and print the result to stdout.
    """
    state = run_query(question)

    print("=" * 60)
    print(f"QUERY: {question}")
    print("-" * 60)

    snippets = state.get("snippets") or []
    if snippets:
        print(f"RETRIEVED {len(snippets)} SNIPPET(S):")
        for snippet in snippets:
            preview = snippet.text[:90].replace("\n", " ")
            print(f"  [{snippet.chunk_id}] score={snippet.score:.3f}  {preview}...")
    else:
        print("RETRIEVED 0 SNIPPETS")

    print("-" * 60)
    print("ANSWER:")
    print(state.get("answer", ""))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="2 Agent RAG over local knowledge base!")
    parser.add_argument("question", nargs="*", help="question to ask; omit to run samples")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug trace")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    load_dotenv()
    warm_index()

    questions = [" ".join(args.question)] if args.question else SAMPLE_QUERIES
    failures = 0
    for question in questions:
        try:
            show(question)
        except Exception as exc:
            failures += 1
            logger.error(
                f"Query failed: {question!r} ({type(exc).__name__}: {exc})",
                exc_info=logger.isEnabledFor(logging.DEBUG)
            )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()