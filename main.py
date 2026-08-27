"""
CLI entry point.

Usage:
    python main.py "Should companies be allowed to use AI to set individual prices?"
    python main.py            # interactive mode
"""
import sys

import config
from graph.debate_graph import build_graph, run_debate


def print_result(state: dict) -> None:
    print("\n" + "=" * 70)
    if state.get("blocked"):
        print("BLOCKED BY INJECTION DEFENSE")
        print(state.get("block_reason", ""))
        print("=" * 70)
        return

    print("TRANSCRIPT")
    print("-" * 70)
    for line in state.get("transcript", []):
        print(line, "\n")

    print("=" * 70)
    print("FINAL SYNTHESIZED ANSWER")
    print("-" * 70)
    print(state.get("final_answer", "(no answer generated)"))
    print("=" * 70 + "\n")


def main():
    if not config.GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        print("Get a free key at https://console.groq.com\n")

    graph = build_graph()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        state = run_debate(query, graph)
        print_result(state)
        return

    print("Cognitive Routing RAG Engine -- interactive mode. Ctrl+C to exit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not query:
            continue
        state = run_debate(query, graph)
        print_result(state)


if __name__ == "__main__":
    main()
