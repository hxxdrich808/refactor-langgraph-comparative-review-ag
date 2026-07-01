"""
Command‑line entry point for the LangGraph Comparative Review Agent.
"""

import argparse
from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, END
from compare_state import CompareState
from nodes import plan_criteria, research_entity, build_table, verdict

# Rich for pretty printing
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Compare three entities.")
    parser.add_argument(
        "--entities",
        nargs="+",
        default=["Chroma", "FAISS", "Qdrant"],
        help="Names of the entities to compare.",
    )
    parser.add_argument(
        "--llm-type",
        choices=["openai", "ollama"],
        default=os.getenv("LLM_TYPE", "openai"),
        help="Which LLM backend to use.",
    )
    parser.add_argument(
        "--enable-qdrant",
        action="store_true",
        help="Enable Qdrant vector storage for caching notes.",
    )
    args = parser.parse_args()

    # Initial state
    initial_state: CompareState = {
        "entities": args.entities,
        "criteria": [],
        "findings": {},
        "final_table": None,
        "verdict": None,
        "llm_type": args.llm_type,
        "use_qdrant": args.enable_qdrant,
        "current_pair_index": 0,
    }

    # Build graph
    workflow = StateGraph(CompareState)
    workflow.add_node("plan_criteria", plan_criteria)
    workflow.add_node("research_entity", research_entity)
    workflow.add_node("build_table", build_table)
    workflow.add_node("verdict", verdict)

    workflow.set_entry_point("plan_criteria")
    workflow.add_edge("plan_criteria", "research_entity")
    # Loop until all pairs processed
    workflow.add_conditional_edges(
        "research_entity",
        lambda state: (
            "build_table" if state["current_pair_index"]
            >= len(state["entities"]) * len(state["criteria"])
            else "research_entity"
        ),
    )
    workflow.add_edge("build_table", "verdict")
    workflow.set_finish_point("verdict")

    graph = workflow.compile()

    # Run
    result = graph(initial_state)

    console = Console()
    if result.get("final_table"):
        console.print(Markdown(result["final_table"]))
    if result.get("verdict"):
        console.print("\n[bold]Verdict:[/]\n" + result["verdict"])

if __name__ == "__main__":
    main()
