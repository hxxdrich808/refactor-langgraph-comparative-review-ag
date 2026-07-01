"""
Entry point for the LangGraph Comparative Review Agent.
"""

import argparse
from compare_state import CompareState
from nodes import (
    plan_criteria,
    research_entity,
    build_table,
    verdict,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Comparative Review Agent")
    parser.add_argument(
        "--entities",
        nargs="+",
        required=True,
        help="List of entities to compare (e.g., Chroma FAISS Qdrant)",
    )
    parser.add_argument(
        "--enable-qdrant",
        action="store_true",
        help="Enable Qdrant vector storage for caching notes",
    )
    parser.add_argument(
        "--llm-type",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    state: CompareState = {
        "entities": args.entities,
        "criteria": [],
        "findings": {},
        "final_table": None,
        "verdict": None,
        "llm_type": args.llm_type,
        "use_qdrant": args.enable_qdrant,
        "current_pair_index": 0,
    }

    # Plan criteria
    state = plan_criteria(state)

    # Research each entity-criterion pair
    total_pairs = len(state["entities"]) * len(state["criteria"])
    while state.get("current_pair_index", 0) < total_pairs:
        state = research_entity(state)

    # Build final table
    state = build_table(state)

    # Generate verdict
    state = verdict(state, args.llm_type)

if __name__ == "__main__":
    main()
